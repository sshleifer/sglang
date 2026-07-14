import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLaunchServer(unittest.TestCase):
    @patch.object(http_server, "_setup_and_run_http_server")
    @patch.object(http_server.Engine, "_launch_subprocesses")
    def test_nonzero_rank_does_not_launch_http_server(
        self, launch_subprocesses, setup_http_server
    ):
        scheduler_init_result = SimpleNamespace(scheduler_infos=[])
        launch_subprocesses.return_value = (
            None,
            None,
            None,
            scheduler_init_result,
            None,
        )

        http_server.launch_server(SimpleNamespace(node_rank=1))

        setup_http_server.assert_not_called()

    @patch.object(http_server, "_setup_and_run_http_server")
    @patch.object(http_server.Engine, "_launch_subprocesses")
    def test_rank_zero_launches_http_server(
        self, launch_subprocesses, setup_http_server
    ):
        scheduler_init_result = SimpleNamespace(scheduler_infos=[])
        launch_subprocesses.return_value = (
            "tokenizer_manager",
            "template_manager",
            "port_args",
            scheduler_init_result,
            "subprocess_watchdog",
        )
        server_args = SimpleNamespace(node_rank=0)

        http_server.launch_server(server_args)

        setup_http_server.assert_called_once_with(
            server_args,
            "tokenizer_manager",
            "template_manager",
            "port_args",
            [],
            "subprocess_watchdog",
            execute_warmup_func=http_server._execute_server_warmup,
            launch_callback=None,
        )


if __name__ == "__main__":
    unittest.main()
