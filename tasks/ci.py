"""
ci tasks
"""
import logging
import sys

import click
from invoke import call, task
from tasks.utils import get_compose_env

from .ml_logger import get_logger  # noqa: E402
from .utils import (
    COLOR_CAUTION,
    COLOR_DANGER,
    COLOR_STABLE,
    COLOR_SUCCESS,
    COLOR_WARNING,
)

LOGGER = get_logger(__name__, provider="Invoke CI", level=logging.INFO)


@task(incrementable=["verbose"])
def clean(ctx, loc="local", verbose=0, cleanup=False):
    """
    clean compiled python artifacts
    Usage: inv ci.clean
    """
    env = get_compose_env(ctx, loc=loc)

    # Override run commands' env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    _cmd = r"""
find . -name '*.pyc' -exec rm -fv {} +
find . -name '*.pyo' -exec rm -fv {} +
find . -name '__pycache__' -exec rm -frv {} +
rm -f .coverage
find . | \
grep -E "(__pycache__|\.pyc$|\.pyo$)" | \
xargs rm -rf
    """

    if verbose >= 1:
        msg = f"{_cmd}"
        click.secho(msg, fg=COLOR_SUCCESS)

    ctx.run(_cmd)


@task(incrementable=["verbose"])
def coverage_clean(ctx, loc="local", verbose=0, cleanup=False):
    """
    clean coverage files
    Usage: inv ci.coverage-clean
    """
    env = get_compose_env(ctx, loc=loc)

    # Override run commands' env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    _cmd = r"""
find . -name '*.pyc' -exec rm -fv {} +
find . -name '*.pyo' -exec rm -fv {} +
find . -name '__pycache__' -exec rm -frv {} +
rm -f .coverage
rm -rf htmlcov/*
rm -rf cov_annotate/*
rm -f cov.xml
    """

    if verbose >= 1:
        msg = f"{_cmd}"
        click.secho(msg, fg=COLOR_SUCCESS)

    ctx.run(_cmd)


@task
def pylint(
    ctx,
    loc="local",
    tests=False,
    everything=False,
    specific="",
    error_only=False,
):
    """
    pylint fastapi_pytorch_postgresql_sandbox folder
    Usage: inv ci.pylint
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    if tests:
        ctx.run(
            "pylint --output-format=colorized --disable=all --enable=F,E --rcfile ./lint-configs-python/python/pylintrc tests",
        )
    elif everything:
        ctx.run(
            "pylint --output-format=colorized --rcfile ./lint-configs-python/python/pylintrc tests fastapi_pytorch_postgresql_sandbox",
        )
    elif specific:
        ctx.run(
            f"pylint --output-format=colorized --disable=all --enable={specific} --rcfile ./lint-configs-python/python/pylintrc tests fastapi_pytorch_postgresql_sandbox",
        )
    elif error_only:
        ctx.run(
            "pylint --output-format=colorized --disable=all --enable=F,E --rcfile ./lint-configs-python/python/pylintrc tests fastapi_pytorch_postgresql_sandbox",
        )
    else:
        ctx.run(
            "pylint --output-format=colorized --disable=all --enable=F,E --rcfile ./lint-configs-python/python/pylintrc fastapi_pytorch_postgresql_sandbox",
        )


@task(incrementable=["verbose"])
def mypy(ctx, loc="local", verbose=0):
    """
    mypy pytorch_lab folder
    Usage: inv ci.mypy
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    # ctx.run("mypy --config-file ./lint-configs-python/python/mypy.ini pytorch_lab tests")
    ctx.run(
        "mypy --config-file ./lint-configs-python/python/mypy.ini fastapi_pytorch_postgresql_sandbox tests",
    )


@task(incrementable=["verbose"])
def sourcery(ctx, loc="local", verbose=0):
    """
    sourcery pytorch_lab folder
    Usage: inv ci.sourcery
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    ctx.run(
        "sourcery review --in-place --enable default --enable google-python-style-guide .",
    )


@task(
    pre=[call(clean, loc="local")],
    incrementable=["verbose"],
)
def black(ctx, loc="local", check=False, debug=False, verbose=0, tests=False):
    """
    Run black code formatter
    Usage: inv ci.black
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    _cmd = "black "

    if check:
        _cmd += "--check "

    if verbose >= 3:
        _cmd += "--verbose "

    if tests:
        _cmd += "tests tasks "

    _cmd += "fastapi_pytorch_postgresql_sandbox"

    if verbose >= 1:
        msg = "[black] bout to run command: \n"
        click.secho(msg, fg="green")
        click.secho(_cmd, fg="green")

    ctx.run(_cmd)


@task(
    pre=[call(clean, loc="local")],
    incrementable=["verbose"],
)
def setup_cfg_fmt(ctx, loc="local", check=False, debug=False, verbose=0, tests=False):
    """
    Run black code formatter
    Usage: inv ci.setup_cfg_fmt
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    _cmd = "setup-cfg-fmt --min-py3-version 3.9 ./setup.cfg "

    if verbose >= 1:
        msg = "[setup-cfg-fmt] bout to run command: \n"
        click.secho(msg, fg="green")
        click.secho(_cmd, fg="green")

    ctx.run(_cmd)


@task(
    pre=[call(clean, loc="local")],
    incrementable=["verbose"],
)
def isort(ctx, loc="local", check=False, dry_run=False, verbose=0, diff=False):
    """
    isort fastapi_pytorch_postgresql_sandbox module. Some of the arguments were taken from the starlette contrib scripts. Tries to align w/ black to prevent having to reformat multiple times.

    To check mode only(does not make changes permenantly):
        Usage: inv ci.isort --check -vvv
    Simply display command we would run:
        Usage: inv ci.isort --check --dry-run -vvv
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    _cmd = "isort "

    if check:
        _cmd += " --check-only"

    if diff:
        _cmd += " --diff"

    if verbose >= 2:
        _cmd += " --verbose"

    _cmd += " fastapi_pytorch_postgresql_sandbox tests"

    if verbose >= 1:
        msg = f"{_cmd}"
        click.secho(msg, fg=COLOR_SUCCESS)

    if dry_run:
        click.secho(
            f"[isort] DRY RUN mode enabled, not executing command: {_cmd}",
            fg=COLOR_CAUTION,
        )
    else:
        ctx.run(_cmd)


@task
def verify_python_version(ctx, loc="local", check=True, debug=False):
    """
    verify_python_version is 3.9.
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    # Python 3.9.0
    res = ctx.run("python --version")

    assert "Python 3.10." in res.stdout.rstrip()


@task
def pre_start(ctx, loc="local", check=True, debug=False):
    """
    pre_start fastapi_pytorch_postgresql_sandbox module
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    # ctx.run("python fastapi_pytorch_postgresql_sandbox/api/tests_pre_start.py")


@task(incrementable=["verbose"])
def pytest(
    ctx,
    loc="local",
    check=True,
    debug=False,
    verbose=0,
    pdb=False,
    mypy=False,
    # configonly=False,
    # settingsonly=False,
    # pathsonly=False,
    workspaceonly=False,
    jsononly=False,
    youtubeonly=False,
    csvonly=False,
    fulltextsearchonly=False,
    txtonly=False,
    # clientonly=False,
    # fastapionly=False,
    # jwtonly=False,
    # mockedfs=False,
    # clionly=False,
    # usersonly=False,
    # convertingtotestclientstarlette=False,
    # loggeronly=False,
    # utilsonly=False,
):
    """
    Run pytest
    Usage: inv ci.pytest
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    _cmd = r"py.test"

    if verbose >= 1:
        msg = "[pytest] check mode disabled"
        click.secho(msg, fg="green")
        _cmd += r" --verbose "

    # if configonly:
    #     _cmd += r" -m configonly "

    # if pathsonly:
    #     _cmd += r" -m pathsonly "

    # if settingsonly:
    #     _cmd += r" -m settingsonly "

    if workspaceonly:
        _cmd += r" -m workspaceonly "

    if jsononly:
        _cmd += r" -m jsononly "

    if youtubeonly:
        _cmd += r" -m youtubeonly "

    if csvonly:
        _cmd += r" -m csvonly "

    if fulltextsearchonly:
        _cmd += r" -m fulltextsearchonly "

    if txtonly:
        _cmd += r" -m txtonly "

    # if clientonly:
    #     _cmd += r" -m clientonly "

    # if fastapionly:
    #     _cmd += r" -m fastapionly "

    # if jwtonly:
    #     _cmd += r" -m jwtonly "

    # if mockedfs:
    #     _cmd += r" -m mockedfs "

    # if clionly:
    #     _cmd += r" -m clionly "

    # if usersonly:
    #     _cmd += r" -m usersonly "

    # if convertingtotestclientstarlette:
    #     _cmd += r" -m convertingtotestclientstarlette "

    # if loggeronly:
    #     _cmd += r" -m loggeronly "

    # if utilsonly:
    #     _cmd += r" -m utilsonly "

    if pdb:
        _cmd += r" --pdb --pdbcls bpdb:BPdb "

    if mypy:
        _cmd += r" --mypy "

    _cmd += r" --cov-config=setup.cfg --verbose --cov-append --cov-report=term-missing --cov-report=xml:cov.xml --cov-report=html:htmlcov --cov-report=annotate:cov_annotate  --showlocals --tb=short --cov=fastapi_pytorch_postgresql_sandbox ."

    resp = ctx.run(_cmd)
    if not resp.ok:
        sys.exit(resp.exited)


@task(incrementable=["verbose"])
def view_coverage(ctx, loc="local"):
    """
    Open coverage report inside of browser
    Usage: inv ci.view-coverage
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    _cmd = r"./scripts/open-browser.py file://${PWD}/htmlcov/index.html"

    ctx.run(_cmd)


@task(
    incrementable=["verbose"],
    aliases=["swagger", "openapi", "view_openapi", "view_swagger"],
)
def view_api_docs(ctx, loc="local"):
    """
    Open api swagger docs inside of browser
    Usage: inv ci.view-api-docs
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    _cmd = r"./script/open-browser.py http://localhost:8000/docs"

    ctx.run(_cmd)


@task(
    incrementable=["verbose"],
    pre=[call(view_api_docs, loc="local"), call(view_coverage, loc="local")],
)
def browser(ctx, loc="local"):
    """
    Open api swagger docs inside of browser
    Usage: inv ci.view-api-docs
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    msg = "Finished loading everything into browser"
    click.secho(msg, fg=COLOR_SUCCESS)


@task(
    pre=[
        call(clean, loc="local"),
        call(verify_python_version, loc="local"),
        call(pre_start, loc="local"),
    ],
    incrementable=["verbose"],
)
def monkeytype(
    ctx,
    loc="local",
    verbose=0,
    cleanup=False,
    test=False,
    apply=False,
    stub=False,
    dry_run=False,
):
    """
    Use monkeytype to collect runtime types of function arguments and return values, and automatically generate stub files
    or even add draft type annotations directly to python code. Uses pytest to access all lines of code that have testing setup.

    To generate stubs:
        Usage: inv ci.monkeytype --test -vvv
    To apply stubs to existing code base:
        Usage: inv ci.monkeytype --test --apply --stub -vvv
    To apply stubs to existing code base(dry run):
        Usage: inv ci.monkeytype --test --apply --stub -vvv --dry-run
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands' env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    if test:
        # NOTE: https://monkeytype.readthedocs.io/en/stable/faq.html#why-did-my-test-coverage-measurement-stop-working
        _cmd = r"""monkeytype run "`command -v pytest`" --no-cov --verbose --mypy --showlocals --tb=short tests"""

        if verbose >= 1:
            msg = f"{_cmd}"
            click.secho(msg, fg=COLOR_SUCCESS)

        if dry_run:
            click.secho(
                f"[monkeytype] DRY RUN mode enabled, not executing command: {_cmd}",
                fg=COLOR_CAUTION,
            )
        else:
            ctx.run(_cmd)

    if stub:
        _cmd_stub = r"""
modules_array=()
while IFS= read -r line; do
    modules_array+=( "$line" )
done < <( monkeytype list-modules | grep -v "pytestipdb" | grep -v "fdf8821871d7_main_tables" | grep -v "env_py" )

echo "Stub all modules using monkeytype"
for element in "${modules_array[@]}"
do
    filename=$(echo $element | sed 's,\.,\/,g')
    _basedir=$(dirname "$filename")
    mkdir -p stubs/$_basedir || true
    touch stubs/$_basedir/__init__.pyi
    echo " [run] monkeytype stub ${element} > stubs/$filename.pyi"
    monkeytype -v stub ${element} > stubs/$filename.pyi
done

    """

        if dry_run:
            click.secho(
                f"[monkeytype] DRY RUN mode enabled, not executing command: \n\n{_cmd_stub}",
                fg=COLOR_CAUTION,
            )
        else:
            ctx.run(_cmd_stub)

    if apply:
        _cmd_apply = r"""
modules_array=()
while IFS= read -r line; do
    modules_array+=( "$line" )
done < <( monkeytype list-modules | grep -v "pytestipdb" | grep -v "fdf8821871d7_main_tables" | grep -v "env_py" )

echo "apply all modules using monkeytype"
for element in "${modules_array[@]}"
do
    monkeytype apply ${element}
done
    """
        if dry_run:
            click.secho(
                f"[monkeytype] DRY RUN mode enabled, not executing command: \n\n{_cmd_apply}",
                fg=COLOR_CAUTION,
            )
        else:
            ctx.run(_cmd_apply)


@task(incrementable=["verbose"])
def autoflake(
    ctx,
    loc="local",
    verbose=0,
    check=False,
    dry_run=False,
    in_place=False,
    remove_all_unused_imports=False,
):
    """
    Use autoflake to remove unused imports, recursively, remove unused variables, and exclude __init__.py

    To run autoflake in check only mode:
        Usage: inv ci.autoflake --check -vvv
    To run autoflake in check only mode(dry-run):
        Usage: inv ci.autoflake --check -vvv --dry-run
    To run autoflake inplace w/(dry-run):
        Usage: inv ci.autoflake --in-place -vvv --dry-run
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands' env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    _cmd = "autoflake" + " --recursive --remove-unused-variables"
    if remove_all_unused_imports:
        _cmd += " --remove-all-unused-imports "

    if check:
        _cmd += " --check"

    if in_place:
        _cmd += " --in-place"

    _cmd += " --exclude=__init__.py"
    _cmd += " fastapi_pytorch_postgresql_sandbox"
    _cmd += " tests"
    _cmd += " tasks"

    if verbose >= 1:
        msg = f"{_cmd}"
        click.secho(msg, fg=COLOR_SUCCESS)

    if dry_run:
        click.secho(
            f"[autoflake] DRY RUN mode enabled, not executing command: {_cmd}",
            fg=COLOR_CAUTION,
        )
    else:
        ctx.run(_cmd)


@task(
    pre=[call(clean, loc="local"), call(verify_python_version, loc="local")],
    incrementable=["verbose"],
    aliases=["clean_stubs", "clean_monkeytype"],
)
def clean_pyi(ctx, loc="local", verbose=0, dry_run=False):
    """
    Clean all stub files

    To clean stubs:
        Usage: inv ci.clean-pyi -vvv
    To clean stubs(dry run):
        Usage: inv ci.clean-pyi -vvv --dry-run
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands' env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    # NOTE: https://monkeytype.readthedocs.io/en/stable/faq.html#why-did-my-test-coverage-measurement-stop-working
    _cmd = r"""find . -name '*.pyi' -exec rm -fv {} +"""

    if verbose >= 1:
        msg = f"{_cmd}"
        click.secho(msg, fg=COLOR_SUCCESS)

    if dry_run:
        click.secho(
            f"[monkeytype] DRY RUN mode enabled, not executing command: {_cmd}",
            fg=COLOR_CAUTION,
        )
    else:
        ctx.run(_cmd)


@task(
    pre=[
        call(clean, loc="local"),
        call(verify_python_version, loc="local"),
        call(mypy, loc="local"),
        call(autoflake, loc="local", in_place=True),
        call(black, loc="local", check=False, tests=True),
        call(isort, loc="local"),
        call(black, loc="local", check=False, tests=True),
        call(mypy, loc="local"),
        call(pylint, loc="local", everything=True),
        call(pytest, loc="local"),
    ],
    incrementable=["verbose"],
)
def lint(ctx, loc="local", check=True, debug=False, verbose=0):
    """
    Run all static analysis[mypy,autoflake,black,isort,black,mypy,pytest]
    Usage: inv ci.lint
    """
    env = get_compose_env(ctx, loc=loc)

    # Only display result
    ctx.config["run"]["echo"] = True

    # Override run commands env variables one key at a time
    for k, v in env.items():
        ctx.config["run"]["env"][k] = v

    if verbose >= 1:
        msg = "[lint] check mode disabled"
        click.secho(msg, fg="green")
