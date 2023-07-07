import cv2
from matplotlib import pyplot as plt

image1 = cv2.imread('lena512.jpg')
#image1 = cv2.imread('mdb003ll.pgm')




#Image negitivity
# image2 = 255 - image1
# plt.subplot(rows,columns,1),plt.imshow(image1)
# plt.subplot(rows,columns,2),plt.imshow(image2)
# plt.show()



#converting RGB to gray and Binary(with out using in built function)
rows = 1
columns = 3
grayimage=cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
print(grayimage.shape)

[threshold,binaryimage]=cv2.threshold(grayimage,127,255,cv2.THRESH_BINARY)
print(binaryimage.shape)

#plt.subplot(rows,columns,1),plt.imshow(cv2.cvtColor(image1,cv2.COLOR_BGR2RGB))


plt.subplot(rows,columns,1),plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)),plt.title('Original image')
plt.subplot(rows,columns,2),plt.imshow(grayimage),plt.title('grayimage')
plt.subplot(rows,columns,3),plt.imshow(binaryimage),plt.title('Binary image')
#display = [image1, grayimage, binaryimage]
#label=['Original Image','Gray image','binaryimage']
plt.show()
cv2.waitKey(0)


# cv2.waitKey(0)
# cv2.destroyAllWindows()