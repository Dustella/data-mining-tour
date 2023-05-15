clc;
clear;
msgfid=fopen('hidden.txt','r');%打开秘密文件，读入秘密信息
[msg,count]= fread(msgfid);
count =count* 8;
alpha=0.02;
fclose(msgfid);msg=dec2bin(msg)';[len col]=size(msg);
io= imread('lenna.bmp');%读取载体图像
io=double(io)/255;
output =io;
il=io(:,:,1);%取图像的一层来隐藏
T=dctmtx(8);%对图像进行分块
DCTrgb=blkproc(i1,[8 8],'P1* x* P2',T,T');% 对图像分块进行DCT 变换
[row,col]=size(DCTrgb);
row=floor(row/8);
col=floor(col/8);

temp = 0;
for i = 1:count
    if msg(i, 1) == 0

        if DCTrgb(i + 4, i + 1) < DCTrgb(i + 3, i + 2) %选择（5,2)和(4,3)这一对系数
            temp = DCTrgb(i +4, i + 1);
            DCTrgb(i + 4, i + 1) = DCTrgb(i + 3, i + 2);
            DCTrgb(i + 3, i + 2) = temp;
        end
    else
        if DCTrgb(i + 4, i + 1) > DCTrgb(i + 3.i + 2)
            temp = DCTrgb(i + 4, i + 1);
            DCTrgb(i + 4, i + 1) = DCTrgb(i + 3, i + 2);
            DCTrgb(i + 3, i + 2) = temp;
        end
    end

    if DCTrgb(i + 4,i + 1) < DCTrgb(i + 3, i + 2)
        DCTrgb(i + 4,i + 1) = DCTrgb(i + 4.i + 1) - alpha;
    else
        DCTrgb(i + 3, i + 2) = DCTrgb(i + 3, i + 2) - alpha;
    end

end
%将信息写回并保存

wi = blkproc(DCTrgb, [88],'P1 * x * P2',T', T);
output = io;
output(:, :, 1) = wi;
imwrite(output, 'watermarkedlena.bmp');
figure;
subplot(1.2, 1); imshow('lena.bmp'); title('原始图像');
subplot(1, 2, 2); imshow('watermarkedlena.bmp'); title('嵌入水印图像');
function msg_bits = str2bit(msgStr)
    msgBin = de2bi(int8(msgStr), 8, 'left-msb');
    len = size(msgBin, 1) .* size(msgBin, 2);
    msg_bits = reshape(double(msgBin).', len, 1).';
end
