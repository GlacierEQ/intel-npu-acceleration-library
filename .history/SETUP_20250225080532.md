////markdown
# Setup Guide

## Adding Python Scripts Directory to PATH

Some Python packages (e.g., ncv-python-headless, pycocotools, neural-compressor) install scripts to:

```
C:\Users\casey\AppData\Roaming\Python\Python310\Scripts
```

To add this directory to your PATH on Windows:

1. Open **Settings** and search for "Environment Variables" and select "Edit the system environment variables".
2. In the System Properties dialog, click on the **Environment Variables…** button.
3. Under **User variables** (or **System variables** if preferred), select the variable named `PATH` and click **Edit**.
4. Click **New** and add:
   ```
   C:\Users\casey\AppData\Roaming\Python\Python310\Scripts
   ```
5. Click **OK** on all dialogs to apply changes.

This will ensure that executables like `f2py.exe` and `incbench.exe` are available from the command line.

## Additional Step: Update PATH

Please add the following path to your system's PATH environment variable:

```
C:\Users\casey\AppData\Roaming\Python\Python310\Scripts
```

After adding it, restart your terminal and run the tests with:
```
pytest test/python
```

Leverage these steps to ensure all installed scripts (e.g., f2py.exe, incbench.exe) become available.
////
