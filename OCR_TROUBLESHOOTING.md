# OCR Troubleshooting Guide

If you're having issues with the OCR (text detection) feature in Smart Goggles, follow this troubleshooting guide.

## Common OCR Issues

### 1. No Text Detected

**Symptoms:**

- "No text detected" message
- No highlighted regions appearing

**Solutions:**

- Check lighting conditions - ensure text is well-lit
- Adjust distance to text (optimal range is 20-40cm)
- Ensure text is in focus and camera is stable
- Try different angles to reduce glare or shadows
- Increase contrast of text if possible

### 2. Incorrect Text Recognition

**Symptoms:**

- Text is detected but incorrectly recognized
- Gibberish or partial text is read out

**Solutions:**

- Modify `.env` settings: Try setting `OCR_CONFIDENCE_THRESHOLD=30` (lower threshold)
- Try different OCR services: Set `OCR_SERVICE=azure` if you have API keys configured
- Ensure text is clearly visible and not distorted
- For non-English text, check if language support is installed for Tesseract
- Adjust camera settings for better contrast

### 3. Text Detection Too Slow

**Symptoms:**

- Long pause between activating OCR mode and getting results
- System appears to freeze temporarily

**Solutions:**

- Check CPU usage - OCR is processor-intensive
- Lower resolution settings in `.env`
- Disable enhanced preprocessing: Set `OCR_TEXT_PREPROCESSING=False`
- Try changing OCR services if available
- Ensure Tesseract is properly installed

### 4. OCR Mode Exits Too Quickly

**Symptoms:**

- OCR mode activates but switches back to normal mode before you can capture text

**Solutions:**

- This has been fixed in the latest update - OCR mode now remains active for 5-8 seconds
- Hold camera more steadily during OCR processing
- Modify code in main.py to extend OCR mode duration if needed

### 5. Wrong Regions Highlighted as Text

**Symptoms:**

- Non-text areas are highlighted as text regions
- Too many false positives

**Solutions:**

- Increase confidence threshold in `.env`: `OCR_CONFIDENCE_THRESHOLD=60`
- Modify object detection settings for better text feature analysis
- Ensure camera is clean and free of smudges/spots
- Check for patterns that might be confused with text

## Tesseract-Specific Issues

### 1. "Tesseract not found" Error

**Solution:**

1. Ensure Tesseract OCR is installed on your system
2. Set the correct path in `.env`:
   ```
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```
3. Verify the installation with command: `tesseract --version`

### 2. Language Support Issues

**Solution:**

1. Install language data files for Tesseract
2. Set language in ocr_module.py:
   ```python
   text = pytesseract.image_to_string(img, lang='eng+fra')  # English + French
   ```

## Cloud OCR Service Issues

### Azure Vision Issues

**Solutions:**

1. Verify your API key and endpoint in `.env`
2. Check your Azure subscription status and quota
3. Ensure you have internet connectivity
4. Check Azure service status for outages

### AWS Textract Issues

**Solutions:**

1. Verify AWS credentials in `.env`
2. Check IAM permissions for Textract service
3. Verify region settings
4. Monitor AWS service health dashboard

## Advanced Troubleshooting

If issues persist:

1. **Enable Debug Logging**:

   - Set logging level to DEBUG in main.py
   - Check smart_goggles.log for detailed information

2. **Test OCR Components Individually**:

   - Run the OCR module separately with test images
   - Test object detection with known text objects

3. **Check Camera Issues**:

   - Try a different camera
   - Update camera drivers
   - Check camera resolution and focus settings

4. **System Resource Check**:
   - Monitor CPU usage during OCR processing
   - Ensure sufficient memory is available
   - Close other intensive applications

## Getting Help

If you still experience issues after trying these troubleshooting steps:

1. File an issue on the GitHub repository
2. Include:
   - Your system specifications
   - Smart Goggles version
   - Log file contents
   - Description of the issue
   - Steps to reproduce the problem
