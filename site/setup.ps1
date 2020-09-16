# check to see if loopback exists
$loopback_exists = $false
foreach ($adapter in Get-NetAdapter){
    if($adapter.InterfaceDescription -eq "Microsoft KM-TEST Loopback Adapter"){
        $loopback_exists = $true
        Write-Output "The LoopBack Adapter exists, checking for status ..."
        if($adapter.Status -eq "Up"){
            Write-Output "The LoopBack Adapter is up and running."
        }
        else
        {
            Enable-NetAdapter -Name $adapter.Name -Confirm:$False
        }
        break
    }
}

# Create the loopback adapter
if(!$loopback_exists)
{
    # Run the hardware wizard
    # When it opens choose the manual option, then Network adapters
    # Choose Microsoft->Microsoft KM-TEST Loopback Adapter
    Start-Process -Wait hdwwiz.exe

    # Get the Adapter Name (Ethernet ##)
    $adapter = (Get-NetAdapter -InterfaceDescription "Microsoft KM-TEST Loopback Adapter")
    
    # Configure the IP address, subnet mask and gateway (Defaults based on README.md)
    netsh int ip set address $adapter.Name static 10.10.10.10 255.255.255.0 10.10.10.254 

    # Confirm Creation
    if($adapter.InterfaceDescription -eq "Microsoft KM-TEST Loopback Adapter"){
        Write-Output "Setup of LoopBack Adapter complete!"
    }
    else
    {
        Write-Output "Setup of LoopBack Adapter failed."
    }
}

## Import data to Django Models

# Setup New Database
python manage.py migrate
set-location data
python import_data.py .
