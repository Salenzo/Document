function Remove-File {
    param (
        [string] $FileName
    )
    if (Test-Path "$FileName") {
        # FIXME: .git has ReadOnly Attribute, I need recurse
        Remove-Item "$FileName" -Recurse
        return 1
    }
    else {
        return 0
    }
}

function Update-Git {
<#
    .Synopsis
        Update file via git
    .Description
        Update file via git
#>

    param (
        [Parameter(Mandatory=$True)]
        [string]$Url
    )
    while ($Url.EndsWith("/")) {
        $Url = $Url.Substring(0, $Url.Length - 1)
    }

    #Write-Host "[I] Remove old files"
    #$tmp = $Url.Split('/')
    #$target = $tmp[$tmp.Length - 1]
    #Remove-File $target | Out-Null
    Write-Host "[I] start to update $Url"
    git clone --depth 1 "$Url"
    #Start-Sleep 1
    #Remove-File "$target/.git/" | Out-Null
}

@(
    "https://github.com/KevinNote/fsharp-starter"
) | ForEach-Object {
    Update-Git $_
}