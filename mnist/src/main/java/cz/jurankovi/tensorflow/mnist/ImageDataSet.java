package cz.jurankovi.tensorflow.mnist;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

public class ImageDataSet extends DataSet {

	public static final int IMG_MAGIC = 2051;

	private RawImages images;

	public ImageDataSet(String path) {
		super(path);
		images = path.endsWith(".gz") ? loadDataSetGzip(path) : loadDataSet(path);
	}

	public RawImages getRawImages() {
		return images;
	}

	public void print(int n) {
		images.print(n);
	}

	protected RawImages loadDataSet(String path) {
		RawImages images = null;
		try (FileInputStream fis = new FileInputStream(path)) {
			checkMagic(fis, IMG_MAGIC);
			int numItems = readInt(fis);
			int nRows = readInt(fis);
			int nCols = readInt(fis);
			LOG.info(
					String.format("Trying to read %d items (with %d rows and %d columns) ...", numItems, nRows, nCols));
			byte[] imagesRaw = new byte[numItems * nRows * nCols];
			int read = fis.read(imagesRaw);
			images = new RawImages(imagesRaw, numItems, nRows, nCols);
			LOG.info(String.format("Images read (read %d bytes)", read));
		} catch (IOException e) {
			LOG.severe(
					String.format("Something went wrong, cannot load data set '%s'. Cause: %s", path, e.getMessage()));
		}
		return images;
	}

	protected RawImages loadDataSetGzip(String path) {
		RawImages images = null;
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
			images = new RawImages(imagesRaw, numItems, nRows, nCols);
			LOG.info(String.format("Images read (read %d bytes)", read));
		} catch (IOException e) {
			LOG.severe(
					String.format("Something went wrong, cannot load data set '%s'. Cause: %s", path, e.getMessage()));
		}
		return images;
	}

	public class RawImages {
		private final byte[] images;
		private final int imgCnt;
		private final int rowCnt;
		private final int colCnt;

		public RawImages(byte[] images, int imgCnt, int rowCnt, int colCnt) {
			this.images = images;
			this.imgCnt = imgCnt;
			this.rowCnt = rowCnt;
			this.colCnt = colCnt;
		}

		public byte[] getImages() {
			return images;
		}

		public int getImgCnt() {
			return imgCnt;
		}

		public int getRowCnt() {
			return rowCnt;
		}

		public int getColCnt() {
			return colCnt;
		}

		public int[][] get2d() {
			return reshape2d();
		}
		
		public int[][][] get3d() {
			return reshape3d();
		}
		
		private int[][][] reshape3d() {
			int[][][] reshaped = new int[imgCnt][rowCnt][colCnt];
			int idx = 0;
			for (int i = 0; i < imgCnt; i++) {
				for (int j = 0; j < rowCnt; j++) {
					for (int k = 0; k < colCnt; k++) {
						reshaped[i][j][k] = images[idx];
						idx++;
					}
				}
			}
			return reshaped;
		}

		private int[][] reshape2d() {
			int pixelCnt = rowCnt * colCnt;
			int[][] reshaped = new int[imgCnt][pixelCnt];
			for (int i = 0; i < imgCnt; i++) {
				for (int j = 0; j < pixelCnt; j++) {
					reshaped[i][j] = images[i * pixelCnt + j];
				}
			}
			return reshaped;
		}

		public void print(int n) {
			int[][][] imgs3d = reshape3d();
			System.out.println(String.format("First %d images: ", n));
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < imgs3d[0].length; j++) {
					for (int k = 0; k < imgs3d[0][0].length; k++) {
						System.out.print(imgs3d[i][j][k] + " ");
					}
					System.out.println();
				}
				System.out.println("\n");
			}
		}

		public int[] pixelsOf(int n) {
			int[][] imgs2d = reshape2d();
			return imgs2d[n];
		}
		
	}

}
