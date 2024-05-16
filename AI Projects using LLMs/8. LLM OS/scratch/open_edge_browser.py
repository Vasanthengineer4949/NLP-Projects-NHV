import webbrowser

# Setting up the path for Microsoft Edge executable
edge_path = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
webbrowser.register('edge', None, webbrowser.BackgroundBrowser(edge_path))

# Open a new web page, here we open google.com
webbrowser.get('edge').open('https://www.google.com')

if __name__ == "__main__":
    # This function call would normally try to open Microsoft Edge
    # and navigate to Google. This should be run where Python and Edge are installed.
    print("Microsoft Edge has been opened with Google homepage.")