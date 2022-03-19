
# Configuration
$VERSION=$args[0]

# Script path
$Path=$PSScriptRoot

# Loop over months and xv-folds
For($taskid=0; $taskid -le 6; $taskid++)
{
    Write-Host "calmodel for task $taskid"

    # Only one batch
    $cmd = "$Path\calmodel.py -v $VERSION -n 1 -t $taskid -p"

    Write-Host "cmd : " $cmd
    Start-Process -WindowStyle Minimized python $cmd
    #Start-Process -Wait python $cmd
}
