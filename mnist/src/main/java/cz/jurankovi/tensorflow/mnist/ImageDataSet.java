package cz.jurankovi.tensorflow.mnist;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

public class ImageDataSet extends DataSet {

	public static final int IMG_MAGIC = 2051;

	private final byte[][][] images;

	public ImageDataSet(String path) {
		super(path);
		images = path.endsWith(".gz") ? loadDataSetGzip(path) : loadDataSet(path);
	}

	public byte[][][] getImages() {
		return images;
	}

	public void print(int n) {
		System.out.println(String.format("First %d images: ", n));
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < images[0].length; j++) {
				for (int k = 0; k < images[0][0].length; k++) {
					System.out.print(images[i][j][k] + " ");
				}
				System.out.println();
			}
			System.out.println("\n");
		}
	}

	public int[] pixelsOf(int n) {
		int height = images[n].length;
		int width = images[n][0].length;
		int[] pixels = new int[height * width];
		int idx = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				pixels[idx] = images[n][i][j];
				idx++;
			}
		}
		return pixels;
	}

	protected byte[][][] loadDataSet(String path) {
		byte[][][] images = new byte[0][0][0];
		try (FileInputStream fis = new FileInputStream(path)) {
			checkMagic(fis, IMG_MAGIC);
			int numItems = readInt(fis);
			int nRows = readInt(fis);
			int nCols = readInt(fis);
			LOG.info(
					String.format("Trying to read %d items (with %d rows and %d columns) ...", numItems, nRows, nCols));
			byte[] imagesRaw = new byte[numItems * nRows * nCols];
			int read = fis.read(imagesRaw);
			images = reshape(imagesRaw, numItems, nRows, nCols);
			LOG.info(String.format("Images read (read %d bytes)", read));
		} catch (IOException e) {
			LOG.severe(
					String.format("Something went wrong, cannot load data set '%s'. Cause: %s", path, e.getMessage()));
		}
		return images;
	}

	protected byte[][][] loadDataSetGzip(String path) {
		byte[][][] images = new byte[0][0][0];
		try (GZIPInputStream gzip = new GZIPInputStream(new FileInputStream(path), GZIP_BUFF_SIZE)) {
			checkMagic(gzip, IMG_MAGIC);
			int numItems = readInt(gzip);
			int nRows = readInt(gzip);
			int nCols = readInt(gzip);
			LOG.info(String.format("Trying to read %d items ...", numItems));
			byte[] imagesRaw = new byte[numItems * nRows * nCols];
			byte[] buff = new byte[GZIP_BUFF_SIZE];
			int read = 0;
			int rb = 0;
			while ((rb = gzip.read(buff)) > -1) {
				System.arraycopy(buff, 0, imagesRaw, read, rb);
				read += rb;
			}
			images = reshape(imagesRaw, numItems, nRows, nCols);
			LOG.info(String.format("Images read (read %d bytes)", read));
		} catch (IOException e) {
			LOG.severe(
					String.format("Something went wrong, cannot load data set '%s'. Cause: %s", path, e.getMessage()));
		}
		return images;
	}

	private byte[][][] reshape(byte[] raw, int items, int rows, int columns) {
		byte[][][] reshaped = new byte[items][rows][columns];
		int idx = 0;
		for (int i = 0; i < items; i++) {
			for (int j = 0; j < rows; j++) {
				for (int k = 0; k < columns; k++) {
					reshaped[i][j][k] = raw[idx];
					idx++;
				}
			}
		}
		return reshaped;
	}

}
