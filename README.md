# -Image-Segmentation-of-Gastrointestinal-Polyps-Using-ResNet-and-U-Net-Architecture
FULL PAPER ATTACHED

Currently, colorectal cancer is one of the most common cancers in the world, with it contributing to over 10% of all cancers. With colorectal cancer being a significant contributor to the worldwide mortality rate, it is imperative to find effective methods for treatment and detection. Early disease detection has a huge impact on survival from colorectal cancer, and polyp detection is therefore important. Although colonoscopies have seemingly been perceived to search and find a large amount of potentially cancerous polyps, there are still many issues with thorough detection. Several studies have shown that many polyps are often overlooked during colonoscopies, with 38.69% of patients having at least one polyp missed during colonoscopy, along with a general polyp miss rate of 17.24%. Increasing the detection of polyps has been shown to decrease the risk of colorectal cancer. Thus, automatic detection of more polyps at an early stage can play a crucial role in improving both prevention of and survival from colorectal cancer. This convolutional neural network model, trained on an annotated medical dataset(KVAIR-SEG), is able to accurately detect gastrointestinal polyps with a state-of-the-art accuracy of 96.8%. This novel machine learning model built using the fast.ai library will immensely improve the polyp detection accuracy of colonoscopies in the future.

Currently, colorectal cancer is one of the most common cancers in the world, with it contributing to over 10% of all cancers[1]. Colorectal cancer is caused by an abnormal growth of cells either in the human rectum or colon. Although not certain, doctors generally agree that this abnormal growth in cells is due to a change in DNA in colon cells, instructing them to multiply quickly, eventually leading to dangerous tumors[5]. Colorectal cancer typically occurs in older patients but can be found at any age. The cancer usually begins with small clumps of cells called polyps that form inside the colon[5]. Several studies have shown that many polyps are often overlooked during colonoscopies and have reported that 38.69% of patients have at least one polyp missed during colonoscopy, along with a general polyp miss rate of 17.24% [2]. Overall, polyps are typically not dangerous, but some can turn cancerous over time, making it essential for proper removal to negate this possibility. Colorectal cancer patients may experience symptoms such as rectal bleeding, weakness or tiredness, ongoing discomfort in the belly area, losing weight without trying, and a change in bowel habits[5]. Cancerous polyps also may not display these symptoms in the patient at first, making screenings for middle to older adults imperative. Increasing the detection of polyps has been shown to decrease the risk of colorectal cancer[3] This project uses machine learning to optimize the detection of gastrointestinal polyps as a checking tool in the case of human error. This convolutional neural network model, trained on an annotated medical dataset(KVAIR-SEG)[4]. The KVAIR-SEG dataset used for the image segmentation contained 2000 images(1000 base images and 1000 masks). 

Machine learning is a computer science discipline that aims to code computers to operate like humans. Machine learning, especially when using convolutional neural networks, is extremely useful in different medical and computer vision tasks such as image segmentation and classification. Convolution neural networks(CNN) are a type of neural network that operates using convolutional layers within the neural network. Each layer in the neural network possesses different filters that identify the characteristics of what you are identifying, which would, in this case, be the gastrointestinal polyp. The deeper through the neural network the image travels, the more complex filters it encounters within the convolutional layers. This project uses the U-net convolutional neural network architecture. By changing the CNN’s architecture, the modified architecture allows for better resolution on less-trained images, as pooling operations are replaced by upsampling operations[6]. A pre-trained model was also implemented using Resnet-34, which is a CNN model trained on Imagenet, a database with over 14 million images. 

For data preparation, the KVAIR-SEG dataset was imported into Kaggle’s coding environment, with file paths created for both the image and mask folders. After visualizing the image sets to check functionality, a data block was created for image processing. A train-test split was created in the dataset, with 80% of the images being used for training and 20% for testing. Fastai image augmentations were also added to diversify the data the model would be trained on. The rand transform augmentation was used to apply a random transformation on each image. The flip item augmentation was also applied to transform images horizontally from left to right. Due to the fact that the images from a colonoscopy will never all be from the same angle, the augmentations were added to adjust to this problem. The images were also turned into tensors as a batch transformation for the computer to be able to interpret. After initializing the data block, the data would fit with a U-Net model with different Resnet models to find which depth was most effective. The four models used were Resnet-18, Resnet-34, Resnet-50, and Resnet 101. The utilization of these models follow the principle of transfer learning where pretrained models are finetuned to successfully function for specific data. Mean_dice was used as a metric in order to compare to other published papers regarding image segmentation of gastrointestinal polyps. 	

The four Resnet models were trained until accuracy rates plateaued and no further improvement was deemed to be possible. Resnet-34 was found to outscore its counterparts with a peak accuracy of 96.8% at epoch 10. Resnet-50 reached a peak accuracy of 94.6% at epoch 18. Resnet-18 reached a peak accuracy of 94.4% at epoch 38. Resnet-101 reached a peak accuracy of 85.1% at epoch 2. A confusion matrix was also constructed with total pixels correctly identified as the metric for the best-performing resnet-34 model.

Gastrointestinal polyps highlight the efficiency of machine learning approaches in healthcare, which have rapidly evolved.  The significance of accurate polyp detection cannot be overstated, given its direct impact on preventing and managing colorectal cancer, a major contributor to global cancer morbidity and mortality. This image segmentation model will serve as an excellent checking tool for doctors and other healthcare professionals to compensate for the possibility of human error during the process of detecting polyps in colonoscopies. Implementing this new method would not impact hospitals’ finances as it is solely software-based, with no new hardware changes needed. For future works, this project would benefit from creating a user friendly desktop application making usage as convenient as possible for healthcare professionals to utilize. Furthermore, collaboration with medical professionals and institutions for extensive validation studies on diverse patient populations would validate the robustness of the model. Continuous refinement and fine-tuning based on feedback from these studies would contribute to the model's adaptability to different clinical scenarios and ensure its reliability in real-world applications. The Resnet-34 model achieved a state-of-the-art accuracy of 96.8% in identifying gastrointestinal polyps, allowing for thorough segmentation with polyp sizes of different sizes.

GENERAL SCRIPT FOR MODEL LOADING AND TRAINING:


from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Path setup
path_im = Path('/kaggle/input/kvair-seg/Kvasir-SEG/images')
path_lbl = Path('/kaggle/input/kvair-seg/Kvasir-SEG/masks')

# Get image files and masks
fnames = get_image_files(path_im)
label_names = get_image_files(path_lbl)

# Get corresponding mask filenames
get_msk = lambda o: path_lbl/f'{o.stem}{o.suffix}'

# Dice coefficient functions
def mean_dice(preds, targets):
    n_classes = preds.shape[1]  # Assuming preds are one-hot encoded or softmax
    preds_cls = preds.argmax(dim=1)  # Convert from probabilities to class indices
    mean_dice_score = 0.0
    for cls_idx in range(n_classes):
        mean_dice_score += binary_dice_coef(preds_cls, targets, cls_idx)
    return mean_dice_score / n_classes

def binary_dice_coef(preds, targets, cls_idx):
    preds_bin = (preds == cls_idx).float()
    targets_bin = (targets == cls_idx).float()
    intersection = (preds_bin * targets_bin).sum()
    union = preds_bin.sum() + targets_bin.sum()
    dice = 2. * intersection / (union + 1e-8)  # Epsilon to avoid division by zero
    return dice

# Define Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.attention(x) * x

# Modify Unet to include Attention Block
def attention_unet_learner(dls, arch=resnet34, **kwargs):
    learn = unet_learner(dls, arch, **kwargs)
    
    # Find the first convolutional layer in the encoder
    # The `learn.model` will have the U-Net architecture, and we'll add the attention block to the first few convolutional layers
    learn.model[0][0] = AttentionBlock(learn.model[0][0].conv[0].out_channels)
    
    return learn

# Define DataBlock
codes = ['n', 'y']
cancer = DataBlock(
    blocks=(ImageBlock, MaskBlock(codes)),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2),
    get_y=get_msk,
    item_tfms=[Resize(128), FlipItem(p=0.5)],  # Fix resize and flip transformations
    batch_tfms=[Normalize.from_stats(*imagenet_stats), IntToFloatTensor(div_mask=255)]  # Normalize and scale masks
)

# Data loaders
source_p = Path('/kaggle/input/kvair-seg/Kvasir-SEG')
dls = cancer.dataloaders(source_p, bs=2)

# Create learner with attention U-Net
learn = attention_unet_learner(dls, metrics=[mean_dice])

# Train the model
learn.fit_one_cycle(5)

# Visualizing Results (Accuracy/Loss Over Epochs, Confusion Matrix, ROC, etc.)
# Visualize Training Loss
plt.plot(range(len(learn.recorder.losses)), learn.recorder.losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Assuming you have true labels (y_true) and predicted labels (y_pred) from the model
# Get predictions (example)
y_true, y_pred = learn.get_preds()

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true.argmax(dim=1), y_pred.argmax(dim=1))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
