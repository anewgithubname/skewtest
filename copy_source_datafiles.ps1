# Define the source and destination folders
$sourceFolder = "..\data_truncated"
$destinationFolder = "..\new_datatrunc"

# Ensure the destination folder exists
if (!(Test-Path -Path $destinationFolder)) {
    New-Item -ItemType Directory -Path $destinationFolder
}

# Define the text to search for in the beginning of folder names
$searchTextList = @("618",
                "619",
                "620",
                "621",
                "622",
                "295",
                "296",
                "297",
                "298",
                "299",
                "850",
                "851",
                "861",
                "862",
                "863",
                "570",
                "572",
                "573",
                "576",
                "577",
                "266",
                "267",
                "270",
                "275",
                "277",
                "822",
                "823",
                "826",
                "833",
                "835",
                "641",
                "642",
                "643",
                "644",
                "645",
                "310",
                "312",
                "313",
                "314",
                "317",
                "866",
                "867",
                "877",
                "878",
                "879",
                "509",
                "510",
                "511",
                "512",
                "513",
                "246",
                "247",
                "248",
                "250",
                "251",
                "790",
                "791",
                "792",
                "793",
                "794")

# Get all subfolders in the source folder
$subfolders = Get-ChildItem -Path $sourceFolder -Directory

foreach ($folder in $subfolders) {
    # Check if the folder name starts with any text in the search list
    foreach ($searchText in $searchTextList) {
        if ($folder.Name.StartsWith($searchText)) {
            # Define the destination path for the folder
            $destinationPath = Join-Path -Path $destinationFolder -ChildPath $folder.Name
            
            # Copy the folder to the destination folder
            Copy-Item -Path $folder.FullName -Destination $destinationPath -Recurse
            break  # Exit the inner loop once a match is found
        }
    }
}

Write-Host "Subfolders starting with any of the specified texts have been copied to '$destinationFolder'."