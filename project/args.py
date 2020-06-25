from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    # Logistics
    parser.add_argument("--data_root", type=str, default="../../../data/celebA", help="Data root directory")
    parser.add_argument("--log_name", type=str, default="test", help="Log name")
    parser.add_argument("--face_detector_pth", type=str, default="models/resnet50_ft_dims_2048.pth", help="Path of pretrained face detector")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs. Use 0 for CPU mode")
    parser.add_argument("--nb_sanity_val_steps", type=int, default=0, help="Number of val sanity checks before starting training")
    parser.add_argument("--num_plots_per_epoch", type=int, default=5, help="How often to plot in one training epoch")
    parser.add_argument("--num_plot_triplets", type=int, default=4, help="How many image triplets to plot")

    # Preprocessing
    parser.add_argument("--img_size", type=int, default=224, help="Spatial size of training images")
    parser.add_argument("--nc", type=int, default=3, help="The number of channels in input images")
    
    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during training")

    # Optimization
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--lr_gen", type=float, default=0.0002, help="Learning rate of generator network")
    parser.add_argument("--lr_dis", type=float, default=0.0002, help="Learning rate of discriminator network")
    parser.add_argument("--gan_mode", type=str, default="lsgan", help="The type of GAN objective. It currently supports vanilla, lsgan, and wgangp.")
    
    # Generator parameters
    parser.add_argument("--nf", type=int, default=64, help="Number of feature maps in encoders")
    parser.add_argument("--nf_mlp", type=int, default=256, help="Number of feature maps for MLP module, i.e. dimension of FC layers")
    parser.add_argument("--down_class", type=int, default=4, help="How often image is downsampled by half of its size in class encoder")
    parser.add_argument("--down_content", type=int, default=3, help="How often image is downsampled by half of its size in content encoder")
    parser.add_argument("--n_mlp_blks", type=int, default=3, help="Number of FC layers in MLP module")
    parser.add_argument("--n_res_blks", type=int, default=2, help="number of ResBlks in content encoder")
    parser.add_argument("--latent_dim", type=int, default=1024, help="Size of latent class code")

    # Discriminator 
    parser.add_argument("--nfd", type=int, default=64, help="The number of filters in the first conv layer of the discriminator")
    parser.add_argument("--dis_arch", type=str, default="basic", help="The architecture's name: basic | n_layers | pixel")
    parser.add_argument("--n_layers_D", type=int, default=3, help="The number of conv layers in the discriminator; effective when netD=='n_layers'")
    parser.add_argument("--norm", type=str, default="instance", help="The type of normalization layers used in the network, either BN or IN.")
    parser.add_argument("--init_type", type=str, default="normal", help="The name of the initialization method for network weights")
    parser.add_argument("--init_gain", type=float, default=0.02, help="Scaling factor for normal, xavier and orthogonal")

    # Loss weights
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight of vgg perceptual loss")
    parser.add_argument("--gamma", type=float, default=10.0, help="Weight of cycle consistency losses")
    parser.add_argument("--lambda_g", type=float, default=1.0, help="Weight of adversarial loss for generator")
    parser.add_argument("--zeta", type=float, default=0.2, help="Weight of discriminator loss")
    
    # Parameters for successive increase of ID loss weight
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs in which delta=delta_min")
    parser.add_argument("--n_epochs_increase", type=int, default=5, help="Number of epochs over which we linearly increase delta from delta_min to delta_max")
    parser.add_argument("--delta_min", type=float, default=0.0, help="Initial value for delta kept for the first <n_epochs>")
    parser.add_argument("--delta_max", type=float, default=1.0, help="Maximum value for delta")

    args = parser.parse_args()

    if args.batch_size < 2:
        raise IndexError(
            "Batch size must be at least 2 because we need 2 input images."
        )
    if args.batch_size % 2 != 0:
        raise IndexError(
            "Batch size must be divisble by 2 because we feed pairs of images to the network."
        )

    return args
