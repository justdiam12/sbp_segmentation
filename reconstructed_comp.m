epoch_50 = imread("/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/reconstructed_50_Epoch.png");
label = imread("/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/reconstructed_label.png");
test = imread("/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/test_label.png");

fig = figure;
sp1 = subplot(3, 2, 1)
sp1.Position = [0.1350 0.7093 0.435 0.2157];
imagesc(R, T, AMP_DATA)
colormap gray;
axis off
clim([150 200])
text(-0.07, 0.95, 'a)', 'FontSize', 14, 'FontWeight', 'bold', 'Units', 'normalized');

sp2 = subplot(3, 2, 2)
sp2.Position = [0.62 0.7093 0.25 0.2157];
imagesc(R2, T, TEST_DATA)
colormap gray;
axis off
clim([150 200])
text(-0.12, 0.95, 'b)', 'FontSize', 14, 'FontWeight', 'bold', 'Units', 'normalized');

sp3 = subplot(3, 2, 3)
sp3.Position = [0.1350 0.4096 0.435 0.2157];
imagesc(label)
axis off
text(-0.07, 0.95, 'c)', 'FontSize', 14, 'FontWeight', 'bold', 'Units', 'normalized');

sp4 = subplot(3,2,5)
sp4.Position = [0.1350 0.1100 0.435 0.2157];
imagesc(epoch_50)
axis off
text(-0.07, 0.95, 'd)', 'FontSize', 14, 'FontWeight', 'bold', 'Units', 'normalized');

sp5 = subplot(3,2,6)
imagesc(test)
sp5.Position = [0.62 0.1100 0.25 0.2157];
axis off
text(-0.12, 0.95, 'e)', 'FontSize', 14, 'FontWeight', 'bold', 'Units', 'normalized');

ax = axes(fig, 'Visible', 'off');
ax.Title.Visible='on';
ax.XLabel.Visible='on';
ax.YLabel.Visible='on';
ylabel(ax, 'Two-Way Travel Time (\mus)', 'FontSize', 14)
xlabel(ax, 'distance (km)', 'FontSize', 14);
saveas(fig, 'reconstructed_image.png');

