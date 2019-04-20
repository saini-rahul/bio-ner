import unittest
from load_data import load_datasets, DATASETS

class test_load_data(unittest.TestCase):
    def test_all_datasets_train(self):
        ret_sentences, ret_labels = load_datasets()
        self.assertEqual(len(ret_sentences), len(DATASETS))
        self.assertEqual(len(ret_labels), len(DATASETS))
        self.assertTrue( len(ret_sentences[0]) > 0 )
    
    def test_all_datasets_test(self):
        ret_sentences, ret_labels = load_datasets(dataset_split='test')
        self.assertEqual(len(ret_sentences), len(DATASETS))
        self.assertEqual(len(ret_labels), len(DATASETS))
        self.assertTrue( len(ret_sentences[0]) > 0 )

    def test_single_dataset_train(self):
        ret_sentences, ret_labels = load_datasets(dataset_index = 0)
        self.assertEqual(len(ret_sentences), 1)
        self.assertEqual(len(ret_labels), 1)
        self.assertTrue( len(ret_sentences[0]) > 0 )
    
    def test_single_dataset_test(self):
        ret_sentences, ret_labels = load_datasets(dataset_index = 0, dataset_split='test')
        self.assertEqual(len(ret_sentences), 1)
        self.assertEqual(len(ret_labels), 1)
        self.assertTrue( len(ret_sentences[0]) > 0 )

if __name__ == '__main__':
    unittest.main()
