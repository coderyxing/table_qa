# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_download.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.download_ranges import fetch


# RangeTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class RangeTests(unittest.TestCase):
    # 回归用例：test_tls_failure_is_retried_without_disabling_verification
    # 应相等：len(calls)，2
    # 应排除：'-k'，calls[0]
    # 应相等：path.read_bytes()，b'abcd'
    def test_tls_failure_is_retried_without_disabling_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'chunk'; calls=[]
            # 模拟 curl 下载的响应和本地文件，验证重试及区间校验分支。
            # 输入：args（已解析的命令参数）；**kwargs。
            # 相关调用：Path(args[args.index('-D') + 1]).write_text。
            def curl(args,**kwargs):
                calls.append(args)
                if len(calls)==1: return SimpleNamespace(returncode=35)
                Path(args[args.index('-D')+1]).write_text('Content-Range: bytes 0-3/8\r\n')
                Path(args[args.index('-o')+1]).write_bytes(b'abcd')
                return SimpleNamespace(returncode=0)
            with patch('scripts.download_ranges.subprocess.run',side_effect=curl), patch('scripts.download_ranges.time.sleep'):
                fetch('https://example.test/model',0,3,8,path)
            self.assertEqual(len(calls),2)
            self.assertNotIn('-k',calls[0])
            self.assertEqual(path.read_bytes(),b'abcd')
    # 回归用例：test_failed_header_does_not_become_a_valid_cached_chunk
    # 应相等：run.call_count，1
    # 应相等：path.read_bytes()，b'abcd'
    def test_failed_header_does_not_become_a_valid_cached_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'chunk'; path.write_bytes(b'xxxx')
            path.with_suffix('.headers').write_text('HTTP/1.1 500')
            # 模拟 curl 下载的响应和本地文件，验证重试及区间校验分支。
            # 输入：args（已解析的命令参数）；**kwargs。
            # 相关调用：Path(args[args.index('-D') + 1]).write_text。
            def curl(args,**kwargs):
                Path(args[args.index('-D')+1]).write_text('Content-Range: bytes 0-3/8\r\n')
                path.write_bytes(b'abcd'); return SimpleNamespace(returncode=0)
            with patch('scripts.download_ranges.subprocess.run',side_effect=curl) as run:
                fetch('https://example.test/model',0,3,8,path)
            self.assertEqual(run.call_count,1)
            self.assertEqual(path.read_bytes(),b'abcd')
