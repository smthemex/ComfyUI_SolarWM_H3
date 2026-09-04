# ComfyUI_SolarWM_H3
[SolarWM](https://github.com/Junchao-cs/SolarWM):Open Data and Scalable Training for Long-Horizon Video World Models,use this node to run minimaxH3


# Tips
*  更多功能，需要官方项目代码放出  
* orbit_turns=0  forward / 前进， >0 turn left/右转 ，<0 turn right/右转  
* radius 镜头轨迹半径  
* radius_end < radius  为推镜头；radius_end > radius 为拉镜头  

# 1.Installation
```
git clone https://github.com/smthemex/ComfyUI_SolarWM_H3
```

# 2.Checkpoints
MinimaxH3-Lora-comfy:
[SolarWM-H3-ComfyUI](https://huggingface.co/smthem/SolarWM-H3-ComfyUI)
```
├── ComfyUI/models/lora/
        ├──adapter_model_comfy_new.safetensors
```

# 3.Example
![](https://github.com/smthemex/ComfyUI_SolarWM_H3/blob/main/example_workflows/example.png)

# 4.Citation
```
@misc{huang2026solarwmopendatascalable,
      title={SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models}, 
      author={Junchao Huang and Guian Fang and Shengju Qian and Xianghao Kong and Zhuoran Zhao and Wei Huang and Yihua Du and Zixin Zhang and Justin Cui and Yuchao Gu and Yukang Chen and Xinting Hu and Tianyu He and Shaoshuai Shi and Zhuotao Tian and Xin Wang and Mike Zheng Shou and Li Jiang},
      year={2026},
      eprint={2609.02886},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.02886}, 
}
```
