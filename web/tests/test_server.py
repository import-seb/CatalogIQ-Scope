import copy
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location("web_server", Path(__file__).resolve().parents[1] / "server.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DemoSafety(unittest.TestCase):
    def setUp(self):
        self.examples = module.read_examples()

    def test_only_exact_features_receive_saved_predictions(self):
        request = copy.deepcopy(self.examples["request"])
        request["products"][0]["record_id"] = "0001"
        result = module.demo_response(request,self.examples)
        self.assertEqual(result["products"][0]["record_id"],"0001")
        self.assertEqual(result["products"][0]["predictions"]["Segment"]["predicted_label"],"Allergy")
        request["products"][0]["ProductName"] += " different"
        result = module.demo_response(request,self.examples)
        for target in result["products"][0]["predictions"].values():
            self.assertIsNone(target["predicted_label"])
            self.assertIsNone(target["confidence"])
        self.assertEqual(result["products"][1]["predictions"]["Segment"]["status"],"suggestion")

    def test_quarantine_does_not_receive_sample_predictions(self):
        request=copy.deepcopy(self.examples["request"])
        request["products"][0]["quarantined"]=True
        self.assertEqual(module.demo_response(request,self.examples)["products"][0]["predictions"]["Segment"]["status"],"unavailable")

    def test_bad_ids_and_injected_truth_fields_rejected(self):
        for mutate in (lambda r:r["products"][0].update(Segment="Allergy"), lambda r:r["products"][1].update(record_id=r["products"][0]["record_id"]),lambda r:r["products"][0].update(record_id=123)):
            request=copy.deepcopy(self.examples["request"])
            mutate(request)
            with self.assertRaises(ValueError):module.validate_request(request)


class LocalHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=module.create_server(0)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url=f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def test_page_and_health(self):
        with urlopen(self.url+"/") as r:
            self.assertEqual(r.status,200)
            self.assertIn("default-src 'self'",r.headers["Content-Security-Policy"])
        with urlopen(self.url+"/api/health") as r:self.assertEqual(json.load(r)["mode"],"demo")

    def test_post_and_origin_guard(self):
        body=json.dumps(module.read_examples()["request"]).encode()
        with urlopen(Request(self.url+"/api/predict",data=body,headers={"Content-Type":"application/json"})) as r:self.assertEqual(len(json.load(r)["products"]),2)
        with self.assertRaises(HTTPError) as error:urlopen(Request(self.url+"/api/predict",data=body,headers={"Content-Type":"application/json","Origin":"https://untrusted.example"}))
        self.assertEqual(error.exception.code,403)
        error.exception.close()

    def test_source_files_and_traversal_not_served(self):
        for path in ("/server.py","/../server.py","/%2e%2e/server.py"):
            with self.assertRaises(HTTPError) as error:urlopen(self.url+path)
            self.assertEqual(error.exception.code,404)
            error.exception.close()

    def test_host_guard(self):
        with self.assertRaises(HTTPError) as error:urlopen(Request(self.url+"/api/health",headers={"Host":"untrusted.example"}))
        self.assertEqual(error.exception.code,403)
        error.exception.close()


if __name__=="__main__":unittest.main()
