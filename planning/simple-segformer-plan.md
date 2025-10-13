# RGB2X Testing Plan: Colab Inference + Local SegFormer Validation                                         
                                                                                                         
## Overview                                                                                             
                                                                                                         
Test Approach 2: Hybrid RGB2X + SegFormer from research doc to validate RGB2X can improve wallpaper      
visualization for complex room geometries.                                                               
                                                                                                         
## Phase 1: Test Image Collection & Organization (Local)                                                
                                                                                                         
### 1.1 Create Test Dataset Structure                                                              
                                                                                                         
- Create test_images/ directory with 5 subdirectories:                                                   
  - 01-simple-cuboid/ - Basic rectangular room (baseline)                                                
  - 02-l-shaped/ - Non-cuboid room geometry                                                              
  - 03-angled-walls/ - Non-orthogonal walls                                                              
  - 04-alcove/ - Room with recessed areas                                                                
  - 05-irregular/ - Complex architectural features                                                       
- Each subdirectory contains:                                                                            
  - input.jpg - Original room photo                                                                      
  - metadata.json - Image source, expected wall count, notes                                             
                                                                                                         
### 1.2 Source Test Images                                                                         
                                                                                                         
- Find/download 5 room images representing each geometry type                                            
- Resize to reasonable dimensions (800-1200px)                                                           
- Document sources and permissions                                                                       
                                                                                                         
## Phase 2: Colab Batch Processing Notebook                                                    
                                                                                                         
### 2.1 Create New Notebook: RGB2X_Testing_Batch.ipynb                                        
                                                                                                         
Based on existing Normal_Irradiance_Colab.ipynb but enhanced with:                                       
                                                                                                         
Key Features:                                                                                            
- Batch processing for multiple test images                                                              
- Upload test_images directory structure to Colab                                                        
- Generate outputs for each image:                                                                       
  - {image_name}_normal.png - RGB-encoded normal map                                                     
  - {image_name}_normal_raw.npy - Raw normal vectors (H×W×3)                                             
  - {image_name}_irradiance.exr - Linear RGB shadow layer                                                
  - {image_name}_irradiance_preview.png - Visual verification                                            
- Organized output structure matching input hierarchy                                                    
- Batch download as zip file                                                                             
                                                                                                         
Modifications from existing notebook:                                                                    
- Add batch upload cell (Google Drive mount or file upload)                                              
- Loop over test images directory                                                                        
- Save raw numpy arrays (.npy) for local processing                                                      
- Create organized output structure                                                                      
- Add progress tracking and error handling                                                               
                                                                                                         
## Phase 3: Local Testing Infrastructure                                                       
                                                                                                         
### 3.1 Create local_testing/ Module                                              
                                                                                                         
New directory with Python scripts for validation:                                                        
                                                                                                         
Files to create:                                                                                         
                                                                                                         
1. load_rgb2x_outputs.py                                                                                 
  - Load normal maps (.npy format) from Colab                                                            
  - Decode RGB-encoded normals to XYZ vectors                                                            
  - Load irradiance EXR files                                                                            
  - Utility functions for data handling                                                                  
2. segformer_integration.py                                                                              
  - Load/initialize SegFormer model (ADE20K)                                                             
  - Generate semantic wall masks                                                                         
  - Integration with RGB2X normals                                                                       
3. normal_clustering.py                                                                                  
  - Implement Algorithm from research doc (lines 187-251)                                                
  - K-means clustering on normals within SegFormer mask                                                  
  - RANSAC plane fitting per cluster                                                                     
  - Generate Wall objects with:                                                                          
      - normal_vector (unit vector)                                                                      
    - pixel_mask (binary mask)                                                                           
    - corners_2d (boundary points)                                                                       
    - confidence (planarity metric)                                                                      
4. visualization.py                                                                                      
  - Overlay normals as RGB color on original image                                                       
  - Draw cluster boundaries                                                                              
  - Compare RGB2X vs ST-RoomNet side-by-side                                                             
  - Save comparison visualizations                                                                       
5. metrics_validation.py                                                                                 
  - Wall count accuracy                                                                                  
  - Normal vector angular error                                                                          
  - Mask IoU (if ground truth available)                                                                 
  - Edge precision (SSAE metric)                                                                         
6. run_test_suite.py                                                                                     
  - Main orchestration script                                                                            
  - Process all 5 test images                                                                            
  - Generate comparison reports                                                                          
  - Save results to test_results/ directory                                                              
                                                                                                         
### 3.2 Dependencies                                                                  
                                                                                                         
Add to local environment:                                                                                
- transformers (SegFormer)                                                                               
- sklearn (K-means clustering)                                                                           
- scipy (connected components, RANSAC)                                                                   
- opencv-python (image processing)                                                                       
- matplotlib (visualization)                                                                             
                                                                                                         
## Phase 4: Validation Workflow                                                  
                                                                                                         
### 4.1 End-to-End Testing                                                             
                                                                                                         
  1. On Colab: Run RGB2X_Testing_Batch.ipynb                                    
   - Upload 5 test images                                                         
   - Generate normal maps + irradiance                                             
   - Download results as zip                                                      
  2. Locally: Extract results to test_outputs/rgb2x/              
  3. Locally: Run python local_testing/run_test_suite.py              
  - Loads RGB2X outputs                                                                                  
  - Runs SegFormer wall segmentation                                                                     
  - Performs hybrid clustering                                                                           
  - Generates visualizations                                                                             
  - Computes metrics                                                                                     
  - Saves report to test_results/{timestamp}/                                                            
                                                                                                         
### 4.2 Expected Outputs                                                             
                                                                                                         
For each test image:                                                                                     
- {image}_rgb2x_walls.png - Detected wall clusters overlaid                                              
- {image}_normals_visualization.png - Color-coded normal map                                             
- {image}_segformer_mask.png - Semantic wall mask                                                        
- {image}_metrics.json - Quantitative results                                                            
- comparison_report.md - Summary findings                                                                
                                                                                                         
## Phase 5: Documentation & Iteration                                                
                                                                                                         
### 5.1 Research Documentation                                                 
                                                                                                         
Create research/06-rgb2x-testing-results.md:                                                             
- Quantitative metrics per test image                                                                    
- Visual comparisons                                                                                     
- Identified strengths/weaknesses                                                                        
- Recommendations for ST-RoomNet v3 integration                                                          
                                                                                                         
### 5.2 Planning Document                                         
                                                                                                         
Create planning/06-rgb2x-local-integration-plan.md:                                                      
- If results positive: Detailed integration plan                                                         
- Module architecture for ST-RoomNet v3 pipeline                                                         
- Performance optimization strategies                                                                    
                                                                                                         
## Key Differences from Existing Work                              
                                                                                                         
- Existing: Normal_Irradiance_Colab.ipynb extracts shadows for overlay                                   
- New: Batch testing workflow to validate RGB2X for wall detection                                       
- Existing: No local SegFormer integration                                                               
- New: Full Approach 2 implementation (Hybrid RGB2X + SegFormer)                                         
- Existing: Single-image workflow                                                                        
- New: Systematic testing with 5 geometry types + metrics                                                
                                                                                                         
## Success Criteria                                              
                                                                                                         
- 5 test images collected and organized                                                                  
- Colab batch notebook processes all images successfully                                                 
- Local testing scripts run without errors                                                               
- Wall detection results visualized for all 5 images                                                     
- Quantitative metrics computed (wall count, normal accuracy)                                            
- Comparison with ST-RoomNet baseline (if available)                                                     
- Research document summarizing findings