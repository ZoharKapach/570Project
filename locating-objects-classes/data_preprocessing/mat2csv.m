data = load("mpii_human_pose_v1_u12_1.mat")
annolist = data.RELEASE.annolist
filenames = {"filename"}
for i=1:numel(annolist)
    file = annolist(i).image.name;
    filenames = [filenames; file];
end

count = {"count"};
locations = {"locations"};
row = 1;
final_classes = {"classes"}
for i=1:numel(annolist)
    people = annolist(i).annorect;
    counter = 0;
    classes = [];
    coord = [];
%     pointsMap = containers.Map();

    for j=1:numel(people)
        if(isfield(people(j), 'annopoints'))
            if(isfield(people(j).annopoints, 'point'))
                myPoints = people(j).annopoints;

                for k=1:numel(myPoints.point)
                    coord = [coord, [myPoints.point(k).x, myPoints.point(k).y]];
                    classes = [classes, myPoints.point(k).id];
                    
%                     if(isKey(pointsMap, class))
%                         pointsMap(class) = [pointsMap(class), coord];
%                     else
%                         pointsMap(class) = [coord];
%                     end
                    counter = counter + 1;
                end
            else
                myPoints = 0;
            end
        end
    end 
    
%     myKeys = keys(pointsMap);
%     myVals = values(pointsMap);
%     col = 0;
%     for l=1:length(myKeys)
%         item_l = myVals{l};
%         size_item = length(item_l);
%         myID = str2num(myKeys{l}) + 1;
%         if (l == 1)
%             locations(row, myID) = item_l;
%      col = col + 1;
%         else
%             locations(row, myIDm) = item_l;
%             col = col + 1;
%         end     
%     end   
     if(isempty(coord))
         coord = 0;
     end
     if(isempty(classes))
         classes = 0;
     end
     final_classes = [final_classes ; classes];
     locations = [locations; coord];
%     final_classes = [final_classes; classes];
%     final_coord = '[';
%     for l=1:2:length(coord)
%       
%         final_coord = strcat(final_coord, "(", num2str(coord(l)), ",", num2str(coord(l+1)), "),");
%     end
%     final_coord = strcat(final_coord, "]");
%     
%     dlmwrite("test1.csv", final_coord, 'roffset', row, 'coffset', 0, '-append');
   
    row = row + 1;
    count = [count; counter];
end
locations_table = cell2table(locations);
classes_table = cell2table(final_classes);
writetable(classes_table, "classes.csv");
writetable(locations_table, "locations.csv");
% files_table = cell2table(filenames);
% counts_table = cell2table(count)
file_counts = [filenames count];
file_counts_table = cell2table(file_counts);
writetable( file_counts_table,"file_counts.csv");


