MODEL_CONFIG = {
    'unet':{
        'backbone':{
            'name':'simplenet',
            'scale_factor':[16,8,4,2,1],
            'in_channels':[32,64,128,256,512],
            'depth':5
        }
        'decoder':{
            'name':'unet_decoder',
            'scale_factor':[16,8,4,2,1],
            'out_channels':[512,256,128,64,32],
            'depth':5
        }
    }
}