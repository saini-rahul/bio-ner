import unittest
from utilities import * 

class test_load_data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sentences = [['19', 'AB'], ['GENE1']]
        ds1 = [ ['I','am','test'], ['this','is','bogus'] ]
        ds2 = [ ['I','am','test2'], ['this','is','bogus'] ]
        cls.ds_sens = []
        cls.ds_sens.append(ds1)
        cls.ds_sens.append(ds2)
        
        cls.ds_tags = []
        dt1 = [ ['O', 'O', 'B-GENE'] , ['B-CHEM','I-CHEM','O']]
        dt2 = [ ['O', 'O', 'B-GENE2'] , ['B-CHEM','I-CHEM','O']]
        cls.ds_tags.append(dt1)
        cls.ds_tags.append(dt2)

    @classmethod
    def tearDownClass(cls):
        pass     
    
    def test_preprocess_sentences(self):
         out_sen =[ ['DIGITDIGIT', 'AB'], ['GENEDIGIT']]
         ret_sen = preprocess_sentences(self.sentences)
         self.assertCountEqual(ret_sen, out_sen)
    
    def  test_create_vocab_tags(self):
          create_vocab_tags(self.ds_tags)   
          all_t = load_dict ('tag2idx.json')
          t1 = load_dict ('tag2idx0.json')
          t2 = load_dict ('tag2idx1.json')           
          
          self.assertEqual (len(all_t), 6) 
          self.assertEqual (len(t1), 5) 
          self.assertEqual (len(t2), 5)
          
          sens = []
          for s in self.ds_sens:
            sens.extend(s)
            
          create_vocab (sens)
         
          ret_tags = prepare_tags (self.ds_tags)
          
          for i, sen_t in enumerate(self.ds_tags[0]):
                for j, t in enumerate(sen_t):
                    self.assertEqual (ret_tags[0][i][j], t1.get(t))
          
    def test_load_embedding_matrix(self):
        load_embedding_matrix()
        e = np.load(DICT_PATH + 'embedding_matrix.npy')
        self.assertEqual(e.shape[1], 200)

    def test_prepare_model (self):
        
        prepare_model (em, n)

if __name__ == '__main__':
    unittest.main()
