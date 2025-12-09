# Dataset Information and Sources

## Overview

This document provides information about the datasets used for training the Aerial Threat Detection System and guidelines for accessing them.

## Recommended Datasets

### 1. UAV Person Detection Dataset

**Source:** [UAV-Person-3 on Roboflow](https://universe.roboflow.com/militarypersons/uav-person-3)

**Description:**
- Aerial imagery captured from drones
- Multiple altitude perspectives
- Various lighting conditions
- Annotated for person detection

**Classes:**
- Person (general)

**Statistics:**
- Images: 2,000+
- Resolution: 640x640 (typical)
- Annotations: YOLO format

**Download:**
```bash
python download_dataset.py download \
  --api-key YOUR_API_KEY \
  --workspace militarypersons \
  --project uav-person-3 \
  --version 1 \
  --output dataset_uav
```

### 2. Combatant Detection Dataset

**Source:** [Combatant-Dataset on Roboflow](https://universe.roboflow.com/minwoo/combatant-dataset)

**Description:**
- Military personnel in various scenarios
- Combat and non-combat situations
- Ground and aerial perspectives
- Soldier identification focus

**Classes:**
- Soldier/Combatant
- Civilian (some versions)

**Statistics:**
- Images: 1,500+
- Various resolutions
- High-quality annotations

**Download:**
```bash
python download_dataset.py download \
  --api-key YOUR_API_KEY \
  --workspace minwoo \
  --project combatant-dataset \
  --version 1 \
  --output dataset_combatant
```

### 3. Soldiers Detection Dataset

**Source:** [Soldiers-Detection-SPF on Roboflow](https://universe.roboflow.com/xphoenixua-nlncq/soldiers-detection-spf)

**Description:**
- Specialized soldier detection
- Multiple terrain types
- Various military uniforms
- High-quality aerial imagery

**Classes:**
- Soldier
- Military vehicle (some versions)

**Statistics:**
- Images: 1,000+
- Mixed resolutions
- Comprehensive annotations

**Download:**
```bash
python download_dataset.py download \
  --api-key YOUR_API_KEY \
  --workspace xphoenixua-nlncq \
  --project soldiers-detection-spf \
  --version 1 \
  --output dataset_soldiers
```

### 4. Look Down Folks Dataset

**Source:** [Look-Down-Folks on Roboflow](https://universe.roboflow.com/folks/look-down-folks)

**Description:**
- Top-down aerial views
- Civilian crowd detection
- Multiple scales and densities
- Various outdoor environments

**Classes:**
- Person/Civilian

**Statistics:**
- Images: 800+
- Consistent aerial perspective
- Good for civilian detection

**Download:**
```bash
python download_dataset.py download \
  --api-key YOUR_API_KEY \
  --workspace folks \
  --project look-down-folks \
  --version 1 \
  --output dataset_folks
```

## Dataset Combination Strategy

### Recommended Approach

For best results, combine multiple datasets to create a diverse training set:

```bash
# Download all datasets
python download_dataset.py download --api-key YOUR_KEY --workspace militarypersons --project uav-person-3 --version 1 --output dataset_uav
python download_dataset.py download --api-key YOUR_KEY --workspace minwoo --project combatant-dataset --version 1 --output dataset_combatant
python download_dataset.py download --api-key YOUR_KEY --workspace xphoenixua-nlncq --project soldiers-detection-spf --version 1 --output dataset_soldiers
python download_dataset.py download --api-key YOUR_KEY --workspace folks --project look-down-folks --version 1 --output dataset_folks

# Combine datasets
python download_dataset.py combine \
  dataset_uav dataset_combatant dataset_soldiers dataset_folks \
  --output combined_dataset
```

### Benefits of Combining Datasets

1. **Increased Diversity:** Different angles, lighting, and scenarios
2. **Better Generalization:** Model learns from varied examples
3. **Improved Accuracy:** More training data leads to better performance
4. **Reduced Bias:** Multiple sources reduce dataset-specific biases

## Dataset Statistics (Combined)

When combining all recommended datasets:

| Split | Approximate Images | Approximate Annotations |
|-------|-------------------|------------------------|
| Train | 3,500-4,500 | 15,000-25,000 |
| Valid | 500-800 | 2,000-4,000 |
| Test | 400-600 | 1,500-3,000 |

**Total:** ~4,500-6,000 images with ~18,000-32,000 annotations

## Data Augmentation

During training, the following augmentations are automatically applied:

### Geometric Augmentations
- **Rotation:** ±10 degrees
- **Translation:** ±10% horizontal/vertical
- **Scale:** 50-150% of original size
- **Horizontal Flip:** 50% probability

### Color Augmentations
- **HSV Adjustments:**
  - Hue: ±1.5%
  - Saturation: ±70%
  - Value: ±40%

### Advanced Augmentations
- **Mosaic:** Combines 4 images into 1
- **MixUp:** Blends two images (optional)
- **Copy-Paste:** Copies objects between images (optional)

## Custom Dataset Creation

### Collecting Your Own Data

If you want to create a custom dataset:

1. **Capture Requirements:**
   - Minimum 500 images per class
   - Various altitudes (50-500m)
   - Different lighting conditions
   - Multiple terrain types
   - Diverse weather (clear, cloudy, etc.)

2. **Image Specifications:**
   - Resolution: Minimum 640x640
   - Format: JPG or PNG
   - Quality: High (avoid compression artifacts)

3. **Annotation Guidelines:**
   - Use tools like [Roboflow](https://roboflow.com) or [CVAT](https://cvat.org)
   - Draw tight bounding boxes
   - Label consistently
   - Include partially visible objects
   - Maintain class balance

### Annotation Tools

#### Roboflow (Recommended)
- Web-based, no installation
- YOLO export built-in
- Smart annotation features
- Version control
- Free tier available

**Website:** https://roboflow.com

#### CVAT
- Open-source
- Advanced features
- Self-hosted or cloud
- Multiple export formats

**Website:** https://cvat.org

#### LabelImg
- Offline tool
- Simple interface
- Direct YOLO format
- Free and open-source

**Installation:**
```bash
pip install labelImg
labelImg
```

## Dataset Quality Checklist

Before training, ensure your dataset meets these criteria:

- [ ] Balanced class distribution (±20% difference)
- [ ] Minimum 500 images per class
- [ ] Clean, accurate annotations
- [ ] No duplicate images
- [ ] Proper train/valid/test split (70/20/10 or 80/15/5)
- [ ] Consistent image quality
- [ ] Diverse scenarios and conditions
- [ ] No corrupted images or labels
- [ ] Proper YOLO format (normalized coordinates)
- [ ] Valid class IDs (0, 1, etc.)

## Data Privacy and Ethics

### Important Considerations

⚠️ **Privacy Notice:**
- Ensure compliance with local privacy laws
- Obtain necessary permissions for data collection
- Anonymize sensitive information
- Follow GDPR/CCPA guidelines if applicable

⚠️ **Ethical Use:**
- This system is for educational purposes
- Military applications require proper oversight
- Consider civilian safety implications
- Avoid bias in data collection and annotation

⚠️ **Data Security:**
- Store datasets securely
- Limit access to authorized personnel
- Use encryption for sensitive data
- Regular backups

## Dataset Licensing

### Roboflow Public Datasets

Most Roboflow Universe datasets are available under various licenses:
- **CC BY 4.0:** Attribution required
- **CC0:** Public domain
- **Custom License:** Check individual dataset

**Always check the specific license for each dataset before use.**

### Your Custom Data

When creating your own dataset:
- Choose an appropriate license
- Document data sources
- Credit contributors
- Specify usage terms

## Troubleshooting Dataset Issues

### Issue: Unbalanced Classes

**Symptoms:** One class has significantly more samples than others

**Solutions:**
- Oversample minority class
- Undersample majority class
- Use class weights during training
- Collect more data for minority class

### Issue: Poor Annotation Quality

**Symptoms:** Incorrect or inconsistent bounding boxes

**Solutions:**
- Review and correct annotations
- Use multiple annotators with consensus
- Implement quality control checks
- Use annotation validation tools

### Issue: Low Dataset Diversity

**Symptoms:** Model performs poorly on real-world data

**Solutions:**
- Add images from different sources
- Include various environmental conditions
- Capture different times of day
- Add challenging examples

### Issue: Corrupted Images or Labels

**Symptoms:** Training crashes or errors

**Solutions:**
```bash
# Verify dataset
python download_dataset.py verify your_dataset/

# Check for corrupted images
python -c "
import cv2
from pathlib import Path
for img in Path('dataset/train/images').glob('*.jpg'):
    if cv2.imread(str(img)) is None:
        print(f'Corrupted: {img}')
"
```

## Additional Resources

### Public Datasets
- [COCO Dataset](https://cocodataset.org/)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [VisDrone Dataset](http://aiskyeye.com/)

### Dataset Tools
- [Roboflow Universe](https://universe.roboflow.com/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Papers With Code Datasets](https://paperswithcode.com/datasets)

### Learning Resources
- [YOLOv8 Dataset Guide](https://docs.ultralytics.com/datasets/)
- [Data Annotation Best Practices](https://roboflow.com/annotate)
- [Computer Vision Dataset Guide](https://www.v7labs.com/blog/computer-vision-datasets)

## Support

For dataset-related questions:
1. Check dataset documentation on Roboflow
2. Review the Training Guide
3. Consult dataset provider forums
4. Create an issue in the repository

---

**Last Updated:** December 2024  
**Maintained by:** CSC Final Project Team
