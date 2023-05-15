%randhiding.m
clc; clear;
fid = fopen('leisheng.wav', 'r');
oa = fread(fid, inf, 'uint8');
fclose(fid);
fid = fopen('hidden.txt', 'r');
[d, n] = fread(fid);
fclose(fid);
d = dec2bin(d, 8);
n = 8 * n;
d = str2num(d(:));
M = oa;

for i = 45:45 + n - 1
    M(i) = bitset(M(i), 1, d(i - 44));
end

figure;
subplot(2, 1, 1);
plot(oa);
title('original audio');
subplot(2, 1, 2);
plot(M);
title('watermarked audio');
fid = fopen('watermarked.wav', 'wb');
fwrite(fid, M, 'uint8');
fclose(fid);
