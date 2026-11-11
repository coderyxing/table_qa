from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.download_ranges import fetch


class RangeTests(unittest.TestCase):
    def test_tls_failure_is_retried_without_disabling_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'chunk'; calls=[]
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
    def test_failed_header_does_not_become_a_valid_cached_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'chunk'; path.write_bytes(b'xxxx')
            path.with_suffix('.headers').write_text('HTTP/1.1 500')
            def curl(args,**kwargs):
                Path(args[args.index('-D')+1]).write_text('Content-Range: bytes 0-3/8\r\n')
                path.write_bytes(b'abcd'); return SimpleNamespace(returncode=0)
            with patch('scripts.download_ranges.subprocess.run',side_effect=curl) as run:
                fetch('https://example.test/model',0,3,8,path)
            self.assertEqual(run.call_count,1)
            self.assertEqual(path.read_bytes(),b'abcd')
