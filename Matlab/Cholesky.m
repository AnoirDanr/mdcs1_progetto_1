% Get full names of all matrices in the directory
matrixFiles = dir('Matrici/*.mtx');
matrixNames = {matrixFiles.name};

% Initialize a cell array to store the results
results = cell(length(matrixNames), 1);

% Loop through each matrix file and read it
for i = 1:length(matrixNames)
    [A,rows,cols,entries,rep,field,symm]  = mmread(fullfile('Matrici', matrixNames{i}));
    xe = ones(1,cols);
    b = A * xe';
    tic;
    R = chol(A);
    chol_time = toc *1000;
    chol_info =  whos('R');
    tic;
    cholTotalBytes = sum([chol_info.bytes]);

    x = R\(R'\b);
    sol_time = toc * 1000;
    total_size=[];
    sol_info = whos('x');
    solTotalBytes = sum([sol_info.bytes]);
    error = norm(x-xe') / norm(xe');
    [m, n] = size(A);
    
    results{i} = {matrixNames{i},m,chol_time,cholTotalBytes / (1024^2), sol_time,solTotalBytes / (1024^2), error};
end




% Prepare the output data
outputData = { 'Matrix Name', 'Matrix size', 'Cholesky Time (ms)', 'Cholesky Memory', 'Solution Time (ms)', 'Solution Memory','Error'; };

% Append results to outputData
for i = 1:length(results)
    outputData = [outputData; ...
                    results{i}];
end

if isunix 
% Define the filename
    filename = 'matlab_linux_output_data.csv';
elseif ispc
    filename = 'matlab_windows_output_data.csv';
end
% Open the file for writing
fileID = fopen(filename, 'w');

% Write the data to the file
for row = 1:size(outputData, 1)
    fprintf(fileID, '%s,', outputData{row, :});
    fprintf(fileID, '\n');
end

% Close the file
fclose(fileID);