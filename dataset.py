from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from glob import glob


def transform(resize = 256, 
              centercrop = 224,
              mean = [0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225]
               ):
    '''
    Standard imagenet transformation
    '''
    transformations = []
    for tr in [transforms.Resize(resize), transforms.CenterCrop(centercrop)]:
      if tr != None:
        transformations.append(tr)
      else:
        pass
    to_tensor = transforms.ToTensor()
    transformations.append(to_tensor)
    if (mean != None) and (std != None):
      normalize = transforms.Normalize(mean=mean, std=std)
      transformations.append(normalize)
    trans = transforms.Compose(transformations)
    return trans

image_transform_ = transform()

def load_image(path, transform = image_transform_):
    '''
    Image loader
    '''
    image = Image.open(path)
    image = transform(image)
    return image


class cat_and_dog(Dataset):
    def __init__(self, path = '../Pictures/PetImages',
                 transform= image_transform_,
                phase = 'train',
                val_portion = 0.2):
        self.path = path 
        self.phase = phase
        self.transform = transform
        if phase != 'test':
            image_list = glob(f'{self.path}/train/*.*')
            random.shuffle(image_list)
            val_split = int(len(image_list)*val_portion)
            val_path = image_list[:val_split]
            train_path = image_list[val_split:]
            self.phase_dict = {'train': train_path, 'val': val_path}
        else:
            image_list = glob(f'{self.path}/{self.phase}/*.*')
            self.phase_dict = {'test':image_list}
        self.labels = {'cat':0, 'dog':1}
        super(cat_and_dog).__init__()
    
    def __len__(self):
        return len(self.phase_dict[self.phase])
    
    def __getitem__(self, idx):
        ids = self.phase_dict[self.phase][idx]
        if self.phase == 'test':
            label = None
        else:
            label = ids.split('/')[-1].split('.')[0]
        image = load_image(ids, self.transform)
        return image, self.labels[label]
   
