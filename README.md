### Installation Steps (Linux Environment)
1. Create a virtual environment (conda or venv)
2. Run `pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com`
3. Run `pip install huggingface_hub`
4. Run `python download_models_hf.py`
5. Navigate to user directory `\home\username`, open `magic-pdf.json` and modify the value of `"device-mode"`
```
{
  "device-mode": "cuda"
}
```
6. Run `pip install -r requirements.txt`
7. Run `python demo.py`

### Installation (Windows Environment)
1. Follow steps on the [mineru repo](https://github.com/opendatalab/MinerU/blob/master/docs/README_Windows_CUDA_Acceleration_en_US.md).
2. Same as steps 6-7 above.