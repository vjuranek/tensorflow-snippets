package cz.jurankovi.tensorflow.mnist;

public class Mnist {

    public static void main(String[] args) throws Exception {
        LabelDataSet labels = new LabelDataSet("/tmp/mnist/data/train-labels-idx1-ubyte.gz", 10);
        ImageDataSet images = new ImageDataSet("/tmp/mnist/data/train-images-idx3-ubyte.gz");
        
        LabelDataSet labelsTest = new LabelDataSet("/tmp/mnist/data/t10k-labels-idx1-ubyte.gz", 10);
        ImageDataSet imagesTest = new ImageDataSet("/tmp/mnist/data/t10k-images-idx3-ubyte.gz");

        /*labels.print(10);
        images.print(1);
        
        
        int[][] imgs = images.getImages().get2d();
        for (int i = 0; i < 10000; i++) {
        	Util.saveAsJPG(28, 28, imgs[i], String.format("/tmp/mnist/jpg/%04d.jpg", i));
        }*/

        
        labelsTest.print(10);
        SoftMax sm = new SoftMax("/tmp/my-model/softmax.pb", labels.getLabels().asOneHotFlatFloat(),
                images.getImages().getImagesAsFloat(), labelsTest.getLabels().asOneHotFlatFloat(),
                imagesTest.getImages().getImagesAsFloat());
        //sm.trainModel();
        sm.loadCheckpoint("/tmp/my-model/softmax.ckpt");
        sm.testModel();
        sm.predictImage(10);

    }

}
