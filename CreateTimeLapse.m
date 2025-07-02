function CreateTimeLapse(app, currentLocation)
    % Create a time-lapse effect by displaying images from a specified folder.
    % Inputs:
    %   app - The application object containing the image display component.
    %   currentLocation - The subfolder name where images are stored.

    % Construct the full path to the image folder
    imageFolder = fullfile(pwd, 'Datasets', currentLocation);

    % Create an imageDatastore to manage the collection of images
    imds = imageDatastore(imageFolder);

    % Get the number of image files in the datastore
    numFiles = numel(imds.Files);

    % Loop through each image file
    for k = 1:numFiles
        % Set the current image as the source for the app's image component
        app.Image.ImageSource = imds.Files{k};

        % Update the figure to display the new image
        drawnow;

        % Pause for 1 second before displaying the next image
        pause(1);
    end
end
