% This was developed for Neural Network B, primarily for debugging Purposes

function void = plotDigit(ioCase)
    x = 1:28;
    y = 1:28; %flip(1:28);
    [X,Y] = meshgrid(x,y);
    %contourf(X,Y,trCase.image,3);
    figure; hold on; xlim([1,28]); ylim([1,28]);
    ax = gca; ax.YDir = 'reverse';
    pbaspect([1 1 1]);
    for r = 1:28
        for c = 1:28
            %RGB = ones(3,1) - (trCase.image(r,c)./256)*ones(3,1); %BLACK on WHITE
            RGB = (ioCase.image(r,c)./256)*ones(3,1); %BLACK on WHITE
            plot(X(r,c), Y(r,c), 'bo', 'MarkerEdgeColor',RGB, 'MarkerFaceColor',RGB);
        end
    end
    title(sprintf('%s Case %d\n', ioCase.purpose, ioCase.i_case));
    hold off;
end