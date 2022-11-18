#数据加载
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor,ToPILImage
from PIL import Image
from glob import glob
import os 
from sklearn.model_selection import train_test_split
from transform import TrainTrainsform,TestTrainsform
from config import data_folder,mask_folder

class SegmentationData(Dataset):
    def __init__(self,data_folder=data_folder,mask_folder=mask_folder,subset="train",trainsform=None):
        super(SegmentationData,self).__init__()
        img_paths = sorted(glob(os.path.join(data_folder,"*.jpg")))
        mask_paths = sorted(glob(os.path.join(mask_folder,"*.jpg")))
        for i in range(len(img_paths)):
            assert os.path.basename(img_paths[i])==os.path.basename(mask_paths[i])
        img_paths_train,img_paths_test,mask_paths_train,mask_paths_test = train_test_split(img_paths,mask_paths,test_size=0.2,random_state=20)
        if subset=="train":
            self.img_paths=img_paths_train
            self.mask_paths = mask_paths_train
        else:
            self.img_paths = img_paths_test
            self.mask_paths = mask_paths_test

        self.transform = trainsform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).resize((224,224))
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path).resize((224,224)).convert("L")
        if self.transform:
            image,mask = self.transform(image,mask)
        else:
            image,mask = ToTensor()(image),ToTensor()(mask)
        return image,mask

    def __len__(self):
        return len(self.img_paths)

if __name__=="__main__":
    topil = ToPILImage()
    transform = TrainTrainsform()
    data = SegmentationData(data_folder,mask_folder,transform)
    image,mask = data[0]
    image,mask = topil(image),topil(mask)
    image.save("./sample.jpg")
    mask.save("./sample_mask.jpg")
    image.show()