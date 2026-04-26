📦 **Overview**

This package provides a comprehensive set of command-line tools for managing files and templates stored in a centralized cloud bucket.
It allows users to efficiently upload, download, list, mark, and manage template files through well-defined commands, each corresponding to a main function defined in the package entry points (e.g., `download_file:main`).

The available commands include:

* **`download`** (`download_file:main`) — Downloads a data from the bucket to the local environment.
* **`upload`** (`upload_file:main`) — Uploads a data file to the bucket.
* **`list`** (`list_files:main`) — Lists all data currently available in the bucket.
* **`show-templates`** (`show_templates:main`) — Displays available templates in the bucket as defined in the configuration file.
* **`break-key`** (`break_file:main_break`) — Marks a specific key in a template as invalid by adding a “?” next to it.
* **`unbreak-key`** (`break_file:main_unbreak`) — Removes the “?” marker from a key in a template, marking it as valid again.
* **`remove`** (`remove_file:main`) — Permanently deletes a key that was previously marked as broken using the `break-key` command.
