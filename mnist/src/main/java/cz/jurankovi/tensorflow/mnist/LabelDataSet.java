package cz.jurankovi.tensorflow.mnist;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

public class LabelDataSet extends DataSet {

    public static final int LABEL_MAGIC = 2049;
    
    private final Labels labels;

    public LabelDataSet(String path, int numClasses) {
        super(path);
        labels = path.endsWith(".gz") ? loadDataSetGzip(path, numClasses) : loadDataSet(path, numClasses);
    }

    public Labels getLabels() {
        return labels;
    }

    public void print(int n) {
        labels.print(n);
    }

    protected Labels loadDataSet(String path, int numClasses) {
        Labels rawLabels = null;
        try (FileInputStream fis = new FileInputStream(path)) {
            checkMagic(fis, LABEL_MAGIC);
            int numItems = readInt(fis);
            LOG.info(String.format("Trying to read %d items ...", numItems));
            byte[] labels = new byte[numItems];
            int read = fis.read(labels, 0, numItems);
            LOG.info(String.format("Labels read (read %d bytes)", read));
            rawLabels = new Labels(labels, numClasses);
        } catch (IOException e) {
            LOG.severe(
                    String.format("Something went wrong, cannot load data set '%s'. Cause: %s", path, e.getMessage()));
        }
        return rawLabels;
    }

    protected Labels loadDataSetGzip(String path, int numClasses) {
        Labels rawLabels = null;
        try (GZIPInputStream gzip = new GZIPInputStream(new FileInputStream(path), GZIP_BUFF_SIZE)) {
            checkMagic(gzip, LABEL_MAGIC);
            int numItems = readInt(gzip);
            LOG.info(String.format("Trying to read %d items ...", numItems));
            byte[] labels = new byte[numItems];
            byte[] buff = new byte[GZIP_BUFF_SIZE];
            int read = 0;
            int rb = 0;
            while ((rb = gzip.read(buff)) > -1) {
                System.arraycopy(buff, 0, labels, read, rb);
                read += rb;
            }
            rawLabels = new Labels(labels, numClasses);
            LOG.info(String.format("Labels read (read %d bytes)", read));
        } catch (IOException e) {
            LOG.severe(
                    String.format("Something went wrong, cannot load data set '%s'. Cause: %s", path, e.getMessage()));
        }
        return rawLabels;
    }

    public class Labels {

        private final byte[] labels;
        private final int numClasses;

        public Labels(byte[] labels, int numClasses) {
            this.labels = labels;
            this.numClasses = numClasses;
        }

        public byte[] getRawLabels() {
            return labels;
        }

        public int getNumClasses() {
            return numClasses;
        }
        
        public int[][] asOneHot() {
            int[][] oneHot = new int[labels.length][numClasses];
            for (int i = 0; i < labels.length; i++) {
                oneHot[i][labels[i]] = 1;
            }
            return oneHot;
        }

        public float[][] asOneHotFloat() {
            float[][] oneHot = new float[labels.length][numClasses];
            for (int i = 0; i < labels.length; i++) {
                oneHot[i][labels[i]] = 1;
            }
            return oneHot;
        }

        public float[] asOneHotFlatFloat() {
            float[] batch = new float[labels.length * numClasses];
            for (int i = 0; i < labels.length; i++) {
                batch[i * numClasses + labels[i]] = 1;
            }
            return batch;
        }

        public float[] getBatch(int n) {
            float[] oneHotFlat = new float[n * numClasses];
            for (int i = 0; i < n; i++) {
                oneHotFlat[i * numClasses + labels[i]] = 1;
            }
            return oneHotFlat;
        }
        
        public void print(int n) {
            System.out.println(String.format("First %d labels: ", n));
            for (int i = 0; i < n; i++) {
                System.out.print(labels[i] + " ");
                if ((i + 1) % 10 == 0) {
                    System.out.println();
                }
            }
        }

    }

}
