import torch
import torch.nn.functional as F
import numpy as np

def Grad_Cam(model, 
             target_layer, 
             image, 
             category):

    def _backward_hook(model, grad_input, grad_output):
        gradients.append(grad_output[0])

    def _forward_hook(model, input, output):
        features.append(output.data)

    features  = []
    gradients = []
    
    backwardHook = target_layer.register_backward_hook(_backward_hook)
    forwardHook  = target_layer.register_forward_hook(_forward_hook)

    output = model(image)

    signal = np.zeros((1, output.size()[-1]), dtype=np.float32)
    signal[0][category] = 1
    signal = torch.from_numpy(signal).requires_grad_(True)
    signal = torch.sum(signal * output)

    model.zero_grad()
    signal.backward(retain_graph=True) 

    gradients = gradients[0][-1].numpy()
    features  = features[0][-1].numpy()

    backwardHook.remove()
    forwardHook.remove()

    return compute_heatmap(features, 
                           np.mean(gradients, axis=(1,2)))


    
def compute_heatmap(features, weights):

    heatmap = np.zeros(features.shape[1:]) 

    for i in range(weights.shape[0]): 
        heatmap += weights[i] * features[i, :, :]

    heatmap = np.maximum(heatmap, 0) 
    heatmap = torch.from_numpy(heatmap.reshape(1,1,7,7))
    heatmap = F.interpolate(heatmap,
                            scale_factor=32,
                            mode='bilinear')
    heatmap = heatmap.numpy()[0,0,:,:]

    return  heatmap / np.max(heatmap)