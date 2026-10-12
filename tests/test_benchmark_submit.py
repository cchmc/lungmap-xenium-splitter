import re
import shlex
from pathlib import Path
from types import SimpleNamespace

from scripts import benchmark_submit


def test_job_script_combines_he_and_verbose_flags_without_blank_argument() -> None:
    script = benchmark_submit._build_job_script(
        job_name="xsplit_sample_with_images",
        dataset_name="sample",
        mode="with_images",
        queue="long",
        project="test",
        cores=1,
        ram_gb="32",
        walltime="01:00",
        log_out="/tmp/job.out",
        log_err="/tmp/job.err",
        env_script=None,
        collector_script="/tmp/benchmark_capture.py",
        metrics_json="/tmp/job.metrics.json",
        splitter_cmd="xenium-splitter",
        input_dir="/data/input",
        lasso_file="/data/regions.csv",
        output_dir="/data/output",
        he_image="/data/he.tif",
        mode_flags="",
        extra_args="",
        verbose_splitter=True,
    )
    command = script.split("\n/usr/bin/time -v ", 1)[1]
    command = command.split("\n\nSPLITTER_EXIT", 1)[0]
    command = re.sub(r"\\\n\s*", " ", command)

    assert shlex.split(command) == [
        "xenium-splitter",
        "split",
        "--input-dir",
        "/data/input",
        "--lasso-file",
        "/data/regions.csv",
        "--output-dir",
        "/data/output",
        "--he-image",
        "/data/he.tif",
        "-v",
    ]
    assert "--he-image '/data/he.tif'" in script.split(
        "'/tmp/benchmark_capture.py'", 1
    )[1]
    assert "--output-json '/tmp/job.metrics.json'" in script
    assert "--exit-code \"$SPLITTER_EXIT\"" in script


def test_job_script_omits_empty_he_capture_argument() -> None:
    script = benchmark_submit._build_job_script(
        job_name="xsplit_sample_data_only",
        dataset_name="sample",
        mode="data_only",
        queue="long",
        project=None,
        cores=1,
        ram_gb="8",
        walltime="01:00",
        log_out="/tmp/job.out",
        log_err="/tmp/job.err",
        env_script=None,
        collector_script="/tmp/benchmark_capture.py",
        metrics_json="/tmp/job.metrics.json",
        splitter_cmd="xenium-splitter",
        input_dir="/data/input",
        lasso_file="/data/regions.csv",
        output_dir="/data/output",
        he_image=None,
        mode_flags="--skip-images",
        extra_args="",
        verbose_splitter=False,
    )

    collector_command = script.split("'/tmp/benchmark_capture.py'", 1)[1]
    collector_command = collector_command.split("\n\necho", 1)[0]
    assert "--he-image" not in collector_command


def test_clear_previous_benchmark_logs_preserves_unrelated_files(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    old_files = [
        "xsplit_old_data_only.out",
        "xsplit_old_data_only.err",
        "xsplit_old_data_only.job",
        "xsplit_old_data_only.metrics.json",
        "benchmark_manifest.csv",
    ]
    for file_name in old_files:
        (log_dir / file_name).write_text("old", encoding="utf-8")
    report_path = log_dir / "report.csv"
    report_path.write_text("keep", encoding="utf-8")
    unrelated_path = log_dir / "notes.txt"
    unrelated_path.write_text("keep", encoding="utf-8")

    removed = benchmark_submit._clear_previous_benchmark_logs(log_dir)

    assert {path.name for path in removed} == set(old_files)
    assert all(not (log_dir / file_name).exists() for file_name in old_files)
    assert report_path.read_text(encoding="utf-8") == "keep"
    assert unrelated_path.read_text(encoding="utf-8") == "keep"


def test_main_creates_per_run_output_directories(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "datasets.csv"
    config_path.write_text(
        "name,input_dir,lasso_file,walltime_data_only,ram_gb_data_only,"
        "walltime_with_images,ram_gb_with_images\n"
        "sample,/data/sample,/data/regions.csv,01:00,8,02:00,16\n",
        encoding="utf-8",
    )
    log_dir = tmp_path / "new" / "logs"
    log_dir.mkdir(parents=True)
    stale_log = log_dir / "xsplit_old_data_only.out"
    stale_log.write_text("old log", encoding="utf-8")
    stale_manifest = log_dir / "benchmark_manifest.csv"
    stale_manifest.write_text("old manifest", encoding="utf-8")
    stale_metrics = log_dir / "xsplit_old_data_only.metrics.json"
    stale_metrics.write_text("{}", encoding="utf-8")
    output_base = tmp_path / "new" / "outputs"
    args = SimpleNamespace(
        config=str(config_path),
        log_dir=str(log_dir),
        output_base=str(output_base),
        dry_run=False,
        timestamp_outputs=False,
        modes="all",
        queue="normal",
        project=None,
        python="xenium-splitter",
        env_script=None,
        extra_args="",
        verbose_splitter=False,
        cores=1,
    )
    monkeypatch.setattr(benchmark_submit, "_parse_args", lambda: args)
    def submit_after_cleanup(script: str, dry_run: bool) -> str:
        assert not stale_log.exists()
        assert not stale_manifest.exists()
        assert not stale_metrics.exists()
        return "123"

    monkeypatch.setattr(benchmark_submit, "_submit_job", submit_after_cleanup)

    benchmark_submit.main()

    assert (output_base / "sample_data_only").is_dir()
    assert (output_base / "sample_with_images").is_dir()
    assert (output_base / "sample_images_only").is_dir()
    images_job = (log_dir / "xsplit_sample_images_only.job").read_text()
    assert "--images-only" in images_job
    assert "--skip-images" not in images_job
    assert "#BSUB -M 16384" in images_job
    assert "#BSUB -W 02:00" in images_job
    assert (log_dir / "benchmark_manifest.csv").is_file()

    config_path.write_text(
        "name,input_dir,lasso_file,walltime_data_only,ram_gb_data_only,"
        "walltime_with_images,ram_gb_with_images,walltime_images_only,ram_gb_images_only\n"
        "sample,/data/sample,/data/regions.csv,01:00,8,02:00,16,03:00,12\n",
        encoding="utf-8",
    )
    args.modes = "images_only"
    benchmark_submit.main()
    images_job = (log_dir / "xsplit_sample_images_only.job").read_text()
    assert "#BSUB -M 12288" in images_job
    assert "#BSUB -W 03:00" in images_job
    assert not (log_dir / "xsplit_sample_data_only.job").exists()
    assert not (log_dir / "xsplit_sample_with_images.job").exists()

    args.modes = "both"
    benchmark_submit.main()
    assert (log_dir / "xsplit_sample_data_only.job").is_file()
    assert (log_dir / "xsplit_sample_with_images.job").is_file()
    assert not (log_dir / "xsplit_sample_images_only.job").exists()