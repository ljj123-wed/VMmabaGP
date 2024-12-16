import numpy as np
from torch.utils.data import DataLoader




class Dendrite_growth_dataLoader(object):
    def __init__(self,mode='train',data_dir='data/Grain_growth',pre_seq_length=10):
        train = np.load(data_dir+'/train.npy')
        train = train[:, :, np.newaxis, :, :]
        self.train_1 = train[:, :pre_seq_length]
        self.train_2 = train[:, pre_seq_length:pre_seq_length*2]
        self.train_3 = train[:, pre_seq_length*2:pre_seq_length*3]
        self.train_4 = train[:, pre_seq_length*3:pre_seq_length*4]
        self.train_5 = train[:, pre_seq_length*4:pre_seq_length*5]



        test = np.load(data_dir+'/test.npy')
        test = test[:, :, np.newaxis, :, :]

        self.test_1 = test[:, :pre_seq_length]
        self.test_2 = test[:, pre_seq_length:pre_seq_length * 2]
        self.test_3 = test[:, pre_seq_length * 2:pre_seq_length * 3]
        self.test_4 = test[:, pre_seq_length * 3:pre_seq_length * 4]
        self.test_5 = test[:, pre_seq_length * 4:pre_seq_length * 5]
        val = np.load(data_dir+'/valid.npy')
        val = val[:, :, np.newaxis, :, :]
        self.val_1 = val[:, :pre_seq_length]
        self.val_2 = val[:, pre_seq_length:pre_seq_length * 2]
        self.val_3 = val[:, pre_seq_length * 2:pre_seq_length * 3]
        self.val_4 = val[:, pre_seq_length * 3:pre_seq_length * 4]
        self.val_5 = val[:, pre_seq_length * 4:pre_seq_length * 5]
        self.mode = mode

    def __len__(self):

        if self.mode == "train":
            return self.train_3.shape[0]
        elif self.mode == 'val':
            return self.val_3.shape[0]
        elif self.mode == 'test':
            return self.test_3.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
          return self.train_1[index],self.train_2[index],self.train_3[index],self.train_4[index],self.train_5[index]
        elif self.mode == "test":
          return self.test_1[index],self.test_2[index],self.test_3[index],self.test_4[index],self.test_5[index]
        elif self.mode == "val":
          return self.val_1[index],self.val_2[index],self.val_3[index],self.val_4[index],self.val_5[index]







class Grain_growth_dataLoader(object):
    def __init__(self,mode='train',data_dir='data/Grain_growth',pre_seq_length=10):
        train = np.load(data_dir+'/train.npy')
        train = train[:, :, np.newaxis, :, :]
        self.train_1 = train[:, :pre_seq_length]
        self.train_2 = train[:, pre_seq_length:pre_seq_length * 2]
       




        test = np.load(data_dir+'/test.npy')
        test = test[:, :, np.newaxis, :, :]

        self.test_1 = test[:, :pre_seq_length]
        self.test_2 = test[:, pre_seq_length:pre_seq_length * 2]
      

        # val = np.load(data_dir+'/valid.npy')
        val = np.load(data_dir+'/Grain_growth200_200.npy')
        
        val = val[:, :, np.newaxis, :, :]
        self.val_1 = val[:, :pre_seq_length]
        self.val_2 = val[:, pre_seq_length:pre_seq_length*2]
        self.val_3 = val[:, pre_seq_length * 2:pre_seq_length * 3]
        self.val_4 = val[:, pre_seq_length * 3:pre_seq_length * 4]
        self.val_5 = val[:, pre_seq_length * 4:pre_seq_length * 5]
        self.val_6 = val[:, pre_seq_length * 5:pre_seq_length * 6]
        self.val_7 = val[:, pre_seq_length * 6:pre_seq_length * 7]
        self.val_8 = val[:, pre_seq_length * 7:pre_seq_length * 8]
        self.val_9 = val[:, pre_seq_length * 8:pre_seq_length * 9]
        self.val_10 = val[:, pre_seq_length * 9:pre_seq_length * 10]
        self.val_11 = val[:, pre_seq_length * 10:pre_seq_length * 11]
        self.val_12 = val[:, pre_seq_length * 11:pre_seq_length * 12]
        self.val_13 = val[:, pre_seq_length * 12:pre_seq_length * 13]
        self.val_14 = val[:, pre_seq_length * 13:pre_seq_length * 14]
        self.val_15 = val[:, pre_seq_length * 14:pre_seq_length * 15]
        self.val_16 = val[:, pre_seq_length * 15:pre_seq_length * 16]
        self.val_17 = val[:, pre_seq_length * 16:pre_seq_length * 17]
        self.val_18 = val[:, pre_seq_length * 17:pre_seq_length * 18]
        self.val_19 = val[:, pre_seq_length * 18:pre_seq_length * 19]
        self.val_20 = val[:, pre_seq_length * 19:pre_seq_length * 20]
     


        self.mode = mode

    def __len__(self):

        if self.mode == "train":
            return self.train_1.shape[0]
        elif self.mode == 'val':
            return self.val_1.shape[0]
        elif self.mode == 'test':
            return self.test_1.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
          return self.train_1[index],self.train_2[index]
        elif self.mode == "test":
          return self.test_1[index],self.test_2[index]
        elif self.mode == "val":
          return self.val_1[index],self.val_2[index],self.val_3[index],self.val_4[index],self.val_5[index],self.val_6[index],self.val_7[index],self.val_8[index],self.val_9[index],self.val_10[index],self.val_11[index],self.val_12[index],self.val_13[index],self.val_14[index],self.val_15[index],self.val_16[index],self.val_17[index],self.val_18[index],self.val_19[index],self.val_20[index]


class Spinodal_decomposition_dataLoader(object):
    def __init__(self,mode='train',data_dir='data/Spinodal_decomposition',pre_seq_length=10):
        train = np.load(data_dir+'/train.npy')
        train = train[:, :, np.newaxis, :, :]
        self.train_1 = train[:, :pre_seq_length]
        self.train_2 = train[:, pre_seq_length:pre_seq_length * 2]
       




        test = np.load(data_dir+'/test.npy')
        test = test[:, :, np.newaxis, :, :]

        self.test_1 = test[:, :pre_seq_length]
        self.test_2 = test[:, pre_seq_length:pre_seq_length * 2]
      

        # val = np.load(data_dir+'/valid.npy')
        val = np.load(data_dir+'/valid.npy')
        
        val = val[:, :, np.newaxis, :, :]
        self.val_1 = val[:, :pre_seq_length]
        self.val_2 = val[:, pre_seq_length:pre_seq_length*2]
        self.val_3 = val[:, pre_seq_length * 2:pre_seq_length * 3]
        self.val_4 = val[:, pre_seq_length * 3:pre_seq_length * 4]
        self.val_5 = val[:, pre_seq_length * 4:pre_seq_length * 5]
        self.val_6 = val[:, pre_seq_length * 5:pre_seq_length * 6]
        self.val_7 = val[:, pre_seq_length * 6:pre_seq_length * 7]
        self.val_8 = val[:, pre_seq_length * 7:pre_seq_length * 8]
        self.val_9 = val[:, pre_seq_length * 8:pre_seq_length * 9]
        self.val_10 = val[:, pre_seq_length * 9:pre_seq_length * 10]
        self.val_11 = val[:, pre_seq_length * 10:pre_seq_length * 11]
        self.val_12 = val[:, pre_seq_length * 11:pre_seq_length * 12]
        self.val_13 = val[:, pre_seq_length * 12:pre_seq_length * 13]
        self.val_14 = val[:, pre_seq_length * 13:pre_seq_length * 14]
        self.val_15 = val[:, pre_seq_length * 14:pre_seq_length * 15]
        self.val_16 = val[:, pre_seq_length * 15:pre_seq_length * 16]
        self.val_17 = val[:, pre_seq_length * 16:pre_seq_length * 17]
        self.val_18 = val[:, pre_seq_length * 17:pre_seq_length * 18]
        self.val_19 = val[:, pre_seq_length * 18:pre_seq_length * 19]
        self.val_20 = val[:, pre_seq_length * 19:pre_seq_length * 20]
     


        self.mode = mode

    def __len__(self):

        if self.mode == "train":
            return self.train_1.shape[0]
        elif self.mode == 'val':
            return self.val_1.shape[0]
        elif self.mode == 'test':
            return self.test_1.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
          return self.train_1[index],self.train_2[index]
        elif self.mode == "test":
          return self.test_1[index],self.test_2[index]
        elif self.mode == "val":
          return self.val_1[index],self.val_2[index],self.val_3[index],self.val_4[index],self.val_5[index],self.val_6[index],self.val_7[index],self.val_8[index],self.val_9[index],self.val_10[index],self.val_11[index],self.val_12[index],self.val_13[index],self.val_14[index],self.val_15[index],self.val_16[index],self.val_17[index],self.val_18[index],self.val_19[index],self.val_20[index]

    
    
class Dendrite_dataLoader(object):
    def __init__(self,mode='train',data_dir='data/Dendrite',pre_seq_length=10):
        train = np.load(data_dir+'/train.npy')
        train = train[:, :, np.newaxis, :, :]/255
        self.train_1 = train[:, :pre_seq_length]
        self.train_2 = train[:, pre_seq_length:pre_seq_length*2]
        self.train_3 = train[:, pre_seq_length*2:pre_seq_length*3]
        self.train_4 = train[:, pre_seq_length*3:pre_seq_length*4]
        self.train_5 = train[:, pre_seq_length*4:pre_seq_length*5]



        test = np.load(data_dir+'/test_0.04-0.06.npy')
        
        test = test[:, :, np.newaxis, :, :]/255

        self.test_1 = test[:, :pre_seq_length]
        self.test_2 = test[:, pre_seq_length:pre_seq_length * 2]
        self.test_3 = test[:, pre_seq_length * 2:pre_seq_length * 3]
        self.test_4 = test[:, pre_seq_length * 3:pre_seq_length * 4]
        self.test_5 = test[:, pre_seq_length * 4:pre_seq_length * 5]
        val = np.load(data_dir+'/valid.npy')
        val = val[:, :, np.newaxis, :, :]/255
        self.val_1 = val[:, :pre_seq_length]
        self.val_2 = val[:, pre_seq_length:pre_seq_length * 2]
        self.val_3 = val[:, pre_seq_length * 2:pre_seq_length * 3]
        self.val_4 = val[:, pre_seq_length * 3:pre_seq_length * 4]
        self.val_5 = val[:, pre_seq_length * 4:pre_seq_length * 5]
        self.mode = mode

    def __len__(self):

        if self.mode == "train":
            return self.train_1.shape[0]
        elif self.mode == 'val':
            return self.val_1.shape[0]
        elif self.mode == 'test':
            return self.test_1.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
          return self.train_1[index],self.train_2[index],self.train_3[index],self.train_4[index],self.train_5[index]
        elif self.mode == "test":
          return self.test_1[index],self.test_2[index],self.test_3[index],self.test_4[index],self.test_5[index]
        elif self.mode == "val":
          return self.val_1[index],self.val_2[index],self.val_3[index],self.val_4[index],self.val_5[index]



class p_dataLoader(object):
    def __init__(self,mode='train',data_dir='data/data_p',pre_seq_length=10):
        train_array = np.load(data_dir+'/train_array.npy')
        train_array = train_array[:, :, np.newaxis, :, :]
        
        test_array = np.load(data_dir+'/test_array.npy')
        test_array = test_array[:, :, np.newaxis, :, :]
        
        
        validation_array = np.load(data_dir+'/validation_array.npy')
        validation_array = validation_array[:, :, np.newaxis, :, :]

        
        self.train_1 = train_array[:,pre_seq_length:pre_seq_length*2:2]
        self.train_2 = train_array[:, pre_seq_length*2:pre_seq_length*3:2]
        self.train_3 = train_array[:, pre_seq_length*3:pre_seq_length*4:2]



        self.val_1 = validation_array[:, pre_seq_length:pre_seq_length*2:2]
        self.val_2 = validation_array[:, pre_seq_length*2:pre_seq_length*3:2]
        self.val_3 = validation_array[:, pre_seq_length*3:pre_seq_length*4:2]


        self.test_1 = test_array[:, pre_seq_length:pre_seq_length*2:2]
        self.test_2 = test_array[:, pre_seq_length*2:pre_seq_length*3:2]
        self.test_3 = test_array[:, pre_seq_length*3:pre_seq_length*4:2]
        
        self.mode = mode

    def __len__(self):

        if self.mode == "train":
            return self.train_1.shape[0]
           
        elif self.mode == 'val':
            return self.val_1.shape[0]
        elif self.mode == 'test':
            return self.test_1.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
          return self.train_1[index],self.train_2[index],self.train_3[index]
        elif self.mode == "test":
          return self.test_1[index],self.test_2[index],self.test_3[index]
        elif self.mode == "val":
          return self.val_1[index],self.val_2[index],self.val_3[index]

    
    
    

    
    
    



def get_loader_segment( batch_size, pre_seq_length=10, mode='train', dataset='KDD'):
    if (dataset == 'Dendrite_growth'):
        dataset = Dendrite_growth_dataLoader(mode=mode,data_dir='data/Dendrite_growth',pre_seq_length=pre_seq_length)
    elif (dataset == 'Grain_growth'):
        dataset = Grain_growth_dataLoader(mode=mode,data_dir='data/Grain_growth',pre_seq_length=pre_seq_length)
    elif (dataset == 'Dendrite'):
        dataset = Dendrite_dataLoader(mode=mode,data_dir='data/Dendrite',pre_seq_length=pre_seq_length)
    elif (dataset =='p'): 
        dataset = p_dataLoader(mode=mode,data_dir='data/data_p',pre_seq_length=pre_seq_length)
    elif (dataset =='Spinodal_decomposition'): 
        dataset = Spinodal_decomposition_dataLoader(mode=mode,data_dir='data/Spinodal_decomposition',pre_seq_length=pre_seq_length)
        
        


    shuffle = False
    if mode == 'train':
        shuffle = True
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8)
    return data_loader

