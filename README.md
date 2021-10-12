# CITGAN
In this work, we propose a novel Cyclic Image Translation Generative Adversarial Network (CIT-GAN) for multi-domain style transfer. To facilitate this, we introduce a Styling Network that has the capability to learn style characteristics of each domain represented in the training dataset. The Styling Network helps the generator to drive the translation of images from a source domain to a reference domain and generate synthetic images with style characteristics of the reference domain. The learned style characteristics for each domain depend on both the style loss and domain classification loss. This induces variability in style characteristics within each domain. The proposed CIT-GAN is used in the context of iris presentation attack detection (PAD) to generate synthetic presentation attack (PA) samples for classes that are under-represented in the training set. 

_S. Yadav and A. Ross, “CIT-GAN: Cyclic Image Translation Generative Adversarial Network With Application in Iris Presentation Attack Detection,” Proc. of IEEE Winter Conference on Applications of Computer Vision (WACV), January 2021._


Clone this repository:

git clone https://github.com/yadavshi/CITGAN.git

cd CITGAN/

Install the dependencies:

conda create -n CITGAN

conda activate CITGAN

conda install -y pytorch torchvision cudatoolkit -c pytorch

conda install x264 ffmpeg -c conda-forge

pip install opencv-python ffmpeg-python scikit-image

pip install pillow scipy tqdm munch

bash download.sh wing

Run following commands:

When Training for the first time:

python main.py --mode train_style --img_size 256 --num_domains 3 --total_iters 300 --batch_size 32 --train_img_dir train_dir --val_img_dir val_dir
               --checkpoint_dir checkpoint_dir

python main.py --mode train --img_size 256 --num_domains 3 --total_iters 100000000 --batch_size 32 --train_img_dir train_dir --val_img_dir val_dir
               --sample_dir ./expr/sample_dir --eval_dir ./expr/eval_dir --result_dir ./expr/result_dir --checkpoint_dir ./expr/checkpoint_dir
(Look into main.py for more arguments' options)

When resuming training:

python main.py --mode train --img_size 256 --num_domains 3 --total_iters 100000000 --batch_size 32 --train_img_dir train_dir --val_img_dir val_dir
               --sample_dir ./expr/sample_dir --eval_dir ./expr/eval_dir --result_dir ./expr/result_dir --checkpoint_dir ./expr/checkpoint_dir --resume_iter 10000

