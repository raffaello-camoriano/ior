function check_output_dir(dir_path)

if ~exist(dir_path,'dir')
    created = mkdir(dir_path);
    if created
        disp(sprintf('Created: %s\n',dir_path));
    else
        error('Failed to create: %s\n',dir_path);
    end
end