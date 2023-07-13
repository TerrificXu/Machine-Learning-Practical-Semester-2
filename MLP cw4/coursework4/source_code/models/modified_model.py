"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch, itertools
from .base_model import BaseModel
from util.image_pool import ImagePool
from . import networks
import numpy as np
import os, cv2, torch
from glob import glob

class ModifiedModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        # if is_train:
        #     parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.

        # return parser
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        visual_names_A = ['real_A', 'fake_A2B', 'rec_A']
        visual_names_B = ['real_B', 'fake_B2A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.

        self.netG_A = networks.UGATIT_ResnetGenerator(input_nc=3, output_nc=3, ngf=opt.ngf, n_blocks=4, img_size=opt.crop_size).to(self.device)
        self.netG_B = networks.UGATIT_ResnetGenerator(input_nc=3, output_nc=3, ngf=opt.ngf, n_blocks=4, img_size=opt.crop_size).to(self.device)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.UGATIT_Discriminator(input_nc = 3, ndf = 64, n_layers=7).to(self.device)
            self.netD_B = networks.UGATIT_Discriminator(input_nc = 3, ndf = 64, n_layers=7).to(self.device)

        self.disLA = networks.UGATIT_Discriminator(input_nc=3, ndf=64, n_layers=5).to(self.device)
        self.disLB = networks.UGATIT_Discriminator(input_nc=3, ndf=64, n_layers=5).to(self.device)

        self.cycle_weight = 10
        self.cam_weight = 1000
        self.identity_weight = 10
        self.result_dir = 'results'
        self.dataset = opt.name

        
        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = networks.RhoClipper(0, 1)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            """ Define Loss """
            self.L1_loss = torch.nn.L1Loss().to(self.device)
            self.MSE_loss = torch.nn.MSELoss().to(self.device)
            self.BCE_loss = torch.nn.BCEWithLogitsLoss().to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            """ Trainer """
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_A2B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_B2A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        #self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B

        self.optimizer_D.zero_grad()

        self.fake_A2B, _, _ = self.netG_A(self.real_A)
        self.fake_B2A, _, _ = self.netG_B(self.real_B)

        real_GA_logit, real_GA_cam_logit, _ = self.netD_A(self.real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(self.real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.netG_B(self.real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(self.real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.netD_A(self.fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(self.fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.netG_B(self.fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(self.fake_A2B)

        D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
        D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
        D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
        D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
        D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
        D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
        D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
        D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

        self.loss_D_A = (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
        self.loss_D_B = (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

        Discriminator_loss = self.loss_D_A + self.loss_D_B
        Discriminator_loss.backward()
        self.optimizer_D.step()

        # Update G
        self.optimizer_G.zero_grad()

        self.fake_A2B, fake_A2B_cam_logit, _ = self.netG_A(self.real_A)
        self.fake_B2A, fake_B2A_cam_logit, _ = self.netG_B(self.real_B)

        self.rec_A, _, _ = self.netG_B(self.fake_A2B)
        self.rec_B, _, _ = self.netG_A(self.fake_B2A)

        self.idt_B, fake_A2A_cam_logit, _ = self.netG_B(self.real_A)
        self.idt_A, fake_B2B_cam_logit, _ = self.netG_A(self.real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.netD_A(self.fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(self.fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.netG_B(self.fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(self.fake_A2B)

        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
        G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
        G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
        G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
        G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

        self.loss_cycle_A = self.L1_loss(self.rec_A, self.real_A)
        self.loss_cycle_B = self.L1_loss(self.rec_B, self.real_B)

        self.loss_idt_B  = self.L1_loss(self.idt_B, self.real_A)
        self.loss_idt_A  = self.L1_loss(self.idt_A, self.real_B)

        G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
        G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

        self.loss_G_A = (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * self.loss_cycle_A + self.identity_weight * self.loss_idt_B  + self.cam_weight * G_cam_loss_A
        self.loss_G_B = (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * self.loss_cycle_B + self.identity_weight * self.loss_idt_A + self.cam_weight * G_cam_loss_B

        Generator_loss = self.loss_G_A + self.loss_G_B
        Generator_loss.backward()
        self.optimizer_G.step()

        # clip parameter of AdaILN and ILN, applied after optimizer step
        self.netG_A.apply(self.Rho_clipper)
        self.netG_B.apply(self.Rho_clipper)

    def RGB2BGR(self, x):
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    def cam(self, x, size = 256):
        x = x - np.min(x)
        cam_img = x / np.max(x)
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, (size, size))
        cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        return cam_img / 255.0
    def tensor2numpy(self, x):
        return x.detach().cpu().numpy().transpose(1,2,0)
    def denorm(self, x):
        return x * 0.5 + 0.5    

    def print(self, inputs, step):
        train_sample_num = 5
        A2B = np.zeros((256 * 7, 0, 3))
        B2A = np.zeros((256 * 7, 0, 3))
        trainA_loader = []
        trainB_loader = []
        for i, data in enumerate(inputs):
            trainA_loader.append(data['A'].to(self.device))
            trainB_loader.append(data['B'].to(self.device))
            if i == train_sample_num:
                break


        self.netG_A.eval(), self.netG_B.eval(), self.netD_A.eval(), self.netD_B.eval(), self.disLA.eval(), self.disLB.eval()
        for i in range(train_sample_num):
            real_A = trainA_loader[i]
            real_B = trainB_loader[i]
            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            fake_A2B, _, fake_A2B_heatmap = self.netG_A(real_A)
            fake_B2A, _, fake_B2A_heatmap = self.netG_B(real_B)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.netG_B(fake_A2B)
            fake_B2A2B, _, fake_B2A2B_heatmap = self.netG_A(fake_B2A)

            fake_A2A, _, fake_A2A_heatmap = self.netG_B(real_A)
            fake_B2B, _, fake_B2B_heatmap = self.netG_A(real_B)

            A2B = np.concatenate((A2B, np.concatenate((self.RGB2BGR(self.tensor2numpy(self.denorm(real_A[0]))),
                                                       self.cam(self.tensor2numpy(fake_A2A_heatmap[0]), 256),
                                                       self.RGB2BGR(self.tensor2numpy(self.denorm(fake_A2A[0]))),
                                                       self.cam(self.tensor2numpy(fake_A2B_heatmap[0]), 256),
                                                       self.RGB2BGR(self.tensor2numpy(self.denorm(fake_A2B[0]))),
                                                       self.cam(self.tensor2numpy(fake_A2B2A_heatmap[0]), 256),
                                                       self.RGB2BGR(self.tensor2numpy(self.denorm(fake_A2B2A[0])))), 0)), 1)

            B2A = np.concatenate((B2A, np.concatenate((self.RGB2BGR(self.tensor2numpy(self.denorm(real_B[0]))),
                                                       self.cam(self.tensor2numpy(fake_B2B_heatmap[0]), 256),
                                                       self.RGB2BGR(self.tensor2numpy(self.denorm(fake_B2B[0]))),
                                                       self.cam(self.tensor2numpy(fake_B2A_heatmap[0]), 256),
                                                       self.RGB2BGR(self.tensor2numpy(self.denorm(fake_B2A[0]))),
                                                       self.cam(self.tensor2numpy(fake_B2A2B_heatmap[0]), 256),
                                                       self.RGB2BGR(self.tensor2numpy(self.denorm(fake_B2A2B[0])))), 0)), 1)
        cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
        cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
        print(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step))
        print('save results')

        self.netG_A.train(), self.netG_B.train(), self.netD_A.train(), self.netD_B.train(), self.disLA.train(), self.disLB.train()
