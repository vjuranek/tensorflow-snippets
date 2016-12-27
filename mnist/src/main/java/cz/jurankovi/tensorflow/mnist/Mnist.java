package cz.jurankovi.tensorflow.mnist;

public class Mnist {
    
    public static final int IMG_SIZE = 28;
    public static final int IMG_PIXELS = IMG_SIZE * IMG_SIZE;
    public static final int NUM_CLASSES = 10;

    public static void main(String[] args) throws Exception {
        LabelDataSet labels = new LabelDataSet("/tmp/mnist/data/train-labels-idx1-ubyte.gz", NUM_CLASSES);
        ImageDataSet images = new ImageDataSet("/tmp/mnist/data/train-images-idx3-ubyte.gz");
        
        LabelDataSet labelsTest = new LabelDataSet("/tmp/mnist/data/t10k-labels-idx1-ubyte.gz", NUM_CLASSES);
        ImageDataSet imagesTest = new ImageDataSet("/tmp/mnist/data/t10k-images-idx3-ubyte.gz");

        /*labels.print(10);
        images.print(1);
        
        
        int[][] imgs = images.getImages().get2d();
        for (int i = 0; i < 10000; i++) {
        	Util.saveAsJPG(28, 28, imgs[i], String.format("/tmp/mnist/jpg/%04d.jpg", i));
        }*/

        
        labelsTest.print(10);
        
        /*SoftMax sm = new SoftMax("/tmp/my-model/softmax.pb", IMG_PIXELS, NUM_CLASSES, labels.getLabels().asOneHotFlatFloat(),
                images.getImages().getImagesAsNormFloat(), labelsTest.getLabels().asOneHotFlatFloat(),
                imagesTest.getImages().getImagesAsNormFloat());
        //sm.trainModel();
        sm.loadCheckpoint("/tmp/my-model/softmax.ckpt");
        sm.testModel();
        sm.predictImage(10);*/ 
        
        NNTwoHiddenLayers nn2h = new NNTwoHiddenLayers("/tmp/my-model/nn2h.pb", IMG_PIXELS, labels.getLabels().asFlatFloat(),
                images.getImages().getImagesAsNormFloat(), labelsTest.getLabels().asFlatFloat(),
                imagesTest.getImages().getImagesAsNormFloat());
        //nn2h.trainModel();
        nn2h.loadCheckpoint("/tmp/my-model/nn2h.ckpt");
        nn2h.testModel();
        nn2h.predictImage(10);

    }

}
