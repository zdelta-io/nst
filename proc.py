import sys
import torch
from torch import optim
from PIL import Image
import torch.nn.functional as Fn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image


# Use Your Own Parameters Instead
num_epochs = 1501
content_weight = 1e1
style_weight = 1e4
content_layer = "conv5_1"

feature_layers = {
    '0': 'conv1_1',
    '5': 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '21': 'conv4_2',
    '28': 'conv5_1'
}

style_layers_dict = {
    'conv1_1': 0.75,
    'conv2_1': 0.5,
    'conv3_1': 0.25,
    'conv4_1': 0.25,
    'conv5_1': 0.25
}


# Use Your Own RGB Settings
H, W = 256, 384
MEAN_RGB = (0.485, 0.456, 0.406)
STD_RGB = (0.229, 0.224, 0.225)


def img_tensor(img_path, device, h=H, w=W,
               mean_rgb=MEAN_RGB, std_rgb=STD_RGB):
    """ Helper Function to get content image tensor and
        style image tensor
    """
    img = Image.open(img_path)
    transformer = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean_rgb, std_rgb)
    ])

    img_tensor = transformer(img)
    # print(img_tensor.shape, img_tensor.requires_grad)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor


def features_extraction(tensor, model, layers):
    """ Util method to fetch the output of intermediate layers
    """
    features = {}
    for name, layer in enumerate(model.children()):
        tensor = layer(tensor)
        if str(name) in layers:
            features[layers[str(name)]] = tensor
    return features


def get_gram_matrix(tensor):
    """ Helper function to compute the gram matrix of a tensor
    """
    n, c, h, w = tensor.size()
    tensor = tensor.view(n * c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def check_content_loss(pred_features, target_features, layer):
    """ Util method to compute content loss
    """
    target = target_features[layer]
    pred = pred_features[layer]
    loss = Fn.mse_loss(pred, target)
    return loss


def check_style_loss(pred_features, target_features,
                     style_layers_dict):
    """ Helper function to compute style loss
    """
    loss = 0
    for layer in style_layers_dict:
        pred_ftr = pred_features[layer]
        pred_gram = get_gram_matrix(pred_ftr)
        n, c, h, w = pred_ftr.shape
        target_gram = get_gram_matrix(target_features[layer])

        layer_loss = style_layers_dict[layer] \
            * Fn.mse_loss(pred_gram, target_gram)
        loss += layer_loss / (n * c * h * w)
    return loss


def img_tensor_to_pil(img_tensor, std_rgb=STD_RGB,
                      mean_rgb=MEAN_RGB):
    """ Helper function to convert tensors back to PIL image
    """
    img_tensor_ = img_tensor.clone().detach()
    img_tensor_ *= torch.tensor(std_rgb).view(3, 1, 1)
    img_tensor_ += torch.tensor(mean_rgb).view(3, 1, 1)
    img_tensor_ = img_tensor_.clamp(0, 1)
    pil_img = to_pil_image(img_tensor_)
    return pil_img


def vgg_model_init():
    """ Using this mothod to get pretrained vgg19 model
    """
    cuda_or_cpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(cuda_or_cpu)

    vgg_model = models.vgg19(pretrained=True) \
        .features \
        .to(device) \
        .eval()

    # print(vgg_model)
    # print(device)
    return device, vgg_model


def main(ctnt_img_path, sty_img_path, output_path):
    """ THE MAIN IMAGE PROCESSING LOGIC HERE
    """
    device, model = vgg_model_init()
    content_tensor = img_tensor(ctnt_img_path, device)
    style_tensor = img_tensor(sty_img_path, device)

    content_features = features_extraction(
        content_tensor,
        model,
        feature_layers)

    style_features = features_extraction(
        style_tensor,
        model,
        feature_layers)

    # Print the shape of content features
    # for key in content_features.keys():
    #    print(content_features[key].shape)

    input_tensor = content_tensor.clone().requires_grad_(True)
    optimizer = optim.Adam([input_tensor], lr=0.01)

    for epoch in range(num_epochs + 1):
        optimizer.zero_grad()
        input_features = features_extraction(
            input_tensor,
            model,
            feature_layers)

        content_loss = check_content_loss(
            input_features,
            content_features,
            content_layer)

        style_loss = check_style_loss(
            input_features,
            style_features,
            style_layers_dict)

        c_loss = content_weight * content_loss
        s_loss = style_weight * style_loss
        neural_loss = c_loss + s_loss

        neural_loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 100 == 0:
            print(
                'Epoch {}, Content loss: {:.2}, style loss {:.2}'
                .format(epoch, content_loss, style_loss))

        img_pil = img_tensor_to_pil(input_tensor[0].cpu())
        img_pil.save(output_path)


if __name__ == "__main__":
    ctnt_img_path = sys.argv[1]
    sty_img_path = sys.argv[2]
    output_path = sys.argv[3]
    main(ctnt_img_path, sty_img_path, output_path)
