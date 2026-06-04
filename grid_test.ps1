# run_grid_search.ps1

# ------------------------------------------------------------------------
# Configuration & Grid Parameters
# ------------------------------------------------------------------------
$models = @("yolov8n", "yolov8s", "yolov8m", "yolov8x")
$confidences = @(0.1, 0.3, 0.5)

# Array to collect tracking metrics for final ranking
$gridResults = @()

Write-Host "Starting Grid Search Exploration..." -ForegroundColor Cyan

# Helper function to assert execution health of native commands
function Check-LastCommand {
    param([string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[ERROR] $StepName failed with exit code $LASTEXITCODE. Terminating pipeline execution to protect GPU compute hours." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ------------------------------------------------------------------------
# Main Pipeline Loop
# ------------------------------------------------------------------------
foreach ($modelPrefix in $models) {
    $modelFile = "${modelPrefix}-seg.pt"

    foreach ($conf in $confidences) {
        # Format folder suffix to replace dots with underscores (e.g., 0.1 becomes 0_1)
        $confStr = $conf.ToString().Replace(".", "_")
        $runSuffix = "${modelPrefix}_${confStr}"

        # Define specialized path outputs for this specific iteration
        $silhouetteDir = "data/silhouettes_$runSuffix"
        $pointCloudDir = "data/point_clouds_$runSuffix"
        $checkpointDir = "checkpoints/scn_$runSuffix"

        Write-Host "`n=========================================================================" -ForegroundColor Cyan
        Write-Host " RUNNING CONFIGURATION: Model = $modelFile | Confidence = $conf" -ForegroundColor Cyan
        Write-Host "=========================================================================" -ForegroundColor Cyan

        # 1) Extract Silhouette Boundaries
        Write-Host "[1/4] Extracting 128-point silhouette boundaries..." -ForegroundColor Yellow
        python extract_silhouettes.py --video_dir data/videos --output_dir $silhouetteDir --boundary_points 128 --model $modelFile --confidence $conf --device cuda
        Check-LastCommand "Silhouette Extraction (extract_silhouettes.py)"

        # 2) Build Stacked Silhouette Point Clouds
        Write-Host "[2/4] Building 60-frame stacked point clouds..." -ForegroundColor Yellow
        python build_point_clouds.py --silhouette_dir $silhouetteDir --output_dir $pointCloudDir --n_points 4096 --temporal_window 60 --boundary_points 128
        Check-LastCommand "Point Cloud Assembly (build_point_clouds.py)"

        # 3) Train SCN Model
        Write-Host "[3/4] Training SCN model (100 epochs)..." -ForegroundColor Yellow
        python train.py --dataset HMDB --data_dir $pointCloudDir --n_classes 2 --epochs 100 --batch_size 64 --output_dir $checkpointDir
        Check-LastCommand "SCN Model Training (train.py)"

        # 4) Evaluate Validation Split
        Write-Host "[4/4] Running model evaluation..." -ForegroundColor Yellow
        $splitFile = "$checkpointDir/splits/test_split1.txt"
        $checkpointFile = "$checkpointDir/best_model.pth"

        # Execute evaluation and capture console output to parse performance metric
        $evalOutput = python evaluate.py --data_dir $pointCloudDir --split $splitFile --checkpoint $checkpointFile --n_classes 2 2>&1 | Out-String
        Check-LastCommand "Validation Evaluation (evaluate.py)"
        
        # Output evaluation log directly to screen for transparency
        Write-Host $evalOutput

        # 5) Parse Performance
        $accuracy = 0.0
        if ($evalOutput -match 'Accuracy:\s*([0-9.]+)') {
            $accuracy = [double]$Matches[1]
        } elseif ($evalOutput -match 'Top-1:\s*([0-9.]+)') {
            $accuracy = [double]$Matches[1]
        } else {
            Write-Host "Warning: Could not automatically parse accuracy. Defaulting metric to 0 for tracking." -ForegroundColor Red
        }

        # Store iteration results
        $gridResults += [PSCustomObject]@{
            Model      = $modelPrefix
            Confidence = $conf
            Accuracy   = $accuracy
        }
    }
}

# ------------------------------------------------------------------------
# Final Ranking Display
# ------------------------------------------------------------------------
Write-Host "`n=========================================================================" -ForegroundColor Green
Write-Host "                         GRID SEARCH COMPLETE                            " -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Green

# Sort metrics in descending order and isolate top 3
$topThree = $gridResults | Sort-Object -Property Accuracy -Descending | Select-Object -First 3

Write-Host "Top 3 Configurations based on performance accuracy:`n" -ForegroundColor Green

# Use a local structural counter loop to output true chronological leaderboard rank
$rankCounter = 1
$displayTable = foreach ($item in $topThree) {
    [PSCustomObject]@{
        Rank       = $rankCounter++
        Model      = $item.Model
        Confidence = $item.Confidence
        Accuracy   = $item.Accuracy
    }
}

$displayTable | Format-Table -Property Rank, Model, Confidence, Accuracy -AutoSize