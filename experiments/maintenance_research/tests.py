"""Model-free tests for maintenance-specific isolation and exact-match behavior."""
import random
import unittest
from core import EXPERIMENT,FIELDS,LIMITS,hash_items
from data import fingerprint,select_groups


class MaintenanceSafety(unittest.TestCase):
    def test_full_feature_fingerprint(self):
        row={name:'item' for name in FIELDS};row['ProductContents']='30 tablets'
        self.assertEqual(fingerprint(row),fingerprint({**row,'Segment':'different label'}))
        self.assertNotEqual(fingerprint(row),fingerprint({**row,'ProductContents':'60 tablets'}))
        self.assertEqual(fingerprint(row),fingerprint({**row,'ProductName':' ITEM '}))

    def test_whole_group_disjoint_proxy_replay(self):
        rows=[{'group_id':str(i//2),'y':i%2} for i in range(40)]
        incoming,replay=select_groups(rows,[i<4 for i in range(40)],incoming_cap=8,replay_cap=12)
        a={rows[i]['group_id'] for i in incoming};b={rows[i]['group_id'] for i in replay}
        self.assertFalse(a&b)
        self.assertEqual((len(incoming),len(replay)),(8,12))
        for group in a|b:self.assertEqual(sum(rows[i]['group_id']==group for i in incoming+replay),2)

    def test_identity_hash_is_order_sensitive(self):
        self.assertNotEqual(hash_items(['1','2']),hash_items(['2','1']))
        self.assertNotEqual(hash_items(['1','23']),hash_items(['12','3']))

    def test_explicit_scope_and_memory_limits(self):
        self.assertFalse(EXPERIMENT['genuine_new_confirmation_data'])
        self.assertFalse(EXPERIMENT['full_retraining'])
        self.assertTrue(EXPERIMENT['human_review_required'])
        self.assertEqual(LIMITS['threads'],4)
        self.assertEqual(LIMITS['worker_bytes'],4*1024**3)
        self.assertLess(LIMITS['allocator_bytes'],LIMITS['used_stop_bytes'])

    def test_gtin_validation_and_sku_scope(self):
        from maintenance_keys import gtin,retailer_sku
        self.assertEqual(gtin('036000291452'),'00036000291452')
        self.assertIsNone(gtin('036000291453'))
        self.assertIsNone(gtin('000000000000'))
        self.assertIsNone(gtin('3.6000291452E10'))
        self.assertNotEqual(retailer_sku({'Retailer':'a','Sku':'1'}),retailer_sku({'Retailer':'b','Sku':'1'}))
        self.assertIsNone(retailer_sku({'Retailer':'a','Sku':''}))

    def test_delta_binding_rejects_wrong_base_and_traversal(self):
        import json,tempfile
        from pathlib import Path
        from core import sha
        from deltas import validate_binding
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);(folder/'head.safetensors').write_bytes(b'unit-fixture')
            source={'files':{'checkpoint.pt':'finished-base'},'metadata':{'target':'Segment','label_names':['a','b']}}
            binding={'base_checkpoint_sha256':'finished-base','target':'Segment','label_names':['a','b'],
                'input_fields':FIELDS,'sequence_length':256,'base_is_existing_fully_fine_tuned_catalogiq_model':True,
                'payload_sha256':{'head.safetensors':sha(folder/'head.safetensors')}}
            path=folder/'maintenance_manifest.json';path.write_text(json.dumps(binding),encoding='utf8')
            self.assertEqual(validate_binding(folder,source)['target'],'Segment')
            path.write_text(json.dumps({**binding,'base_checkpoint_sha256':'wrong-base'}),encoding='utf8')
            with self.assertRaises(RuntimeError):validate_binding(folder,source)
            path.write_text(json.dumps({**binding,'payload_sha256':{'../outside':'bad'}}),encoding='utf8')
            with self.assertRaises(RuntimeError):validate_binding(folder,source)


if __name__=='__main__':unittest.main()
