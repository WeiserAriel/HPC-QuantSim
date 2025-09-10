"""
Data loader for various market data formats in HPC QuantSim.

Supports loading from:
- Parquet files (preferred for performance)
- HDF5 files  
- CSV files
- Arrow/Feather files
- Custom binary formats
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class DataFormat(Enum):
    """Supported data formats."""
    PARQUET = "parquet"
    HDF5 = "hdf5"
    CSV = "csv"
    FEATHER = "feather"
    ARROW = "arrow"
    AUTO = "auto"


class DataLoader:
    """
    High-performance data loader for market data.
    
    Optimized for large datasets with support for:
    - Columnar formats (Parquet, Arrow)
    - Time-based filtering
    - Symbol-based filtering  
    - Memory-efficient streaming
    - Parallel loading for multiple files
    """
    
    def __init__(self):
        """Initialize data loader."""
        self.logger = logging.getLogger(__name__)
        
        # Column name mappings for different data sources
        self.column_mappings = {
            'timestamp': ['timestamp', 'time', 'datetime', 'ts', 'date_time'],
            'symbol': ['symbol', 'ticker', 'instrument', 'sym'],
            'price': ['price', 'last_price', 'close', 'last'],
            'volume': ['volume', 'size', 'quantity', 'vol'],
            'bid_price': ['bid_price', 'bid', 'bp'],
            'ask_price': ['ask_price', 'ask', 'ap'],
            'bid_size': ['bid_size', 'bid_quantity', 'bq'],
            'ask_size': ['ask_size', 'ask_quantity', 'aq'],
            'trade_price': ['trade_price', 'execution_price', 'fill_price'],
            'trade_id': ['trade_id', 'execution_id', 'fill_id'],
        }
        
        # Data type optimizations
        self.dtypes = {
            'price': 'float32',
            'volume': 'float32', 
            'bid_price': 'float32',
            'ask_price': 'float32',
            'bid_size': 'float32',
            'ask_size': 'float32',
            'trade_price': 'float32',
        }
    
    def load(self, data_path: Union[str, Path],
             symbols: Optional[List[str]] = None,
             start_time: Optional[datetime] = None,
             end_time: Optional[datetime] = None,
             data_format: Optional[DataFormat] = None) -> Dict[str, pd.DataFrame]:
        """
        Load market data from file.
        
        Args:
            data_path: Path to data file or directory
            symbols: List of symbols to load (None for all)
            start_time: Start time filter
            end_time: End time filter
            data_format: Data format (auto-detected if None)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Auto-detect format if not specified
        if data_format is None or data_format == DataFormat.AUTO:
            data_format = self._detect_format(data_path)
        
        self.logger.info(f"Loading {data_format.value} data from {data_path}")
        
        # Load data based on format
        if data_format == DataFormat.PARQUET:
            return self._load_parquet(data_path, symbols, start_time, end_time)
        elif data_format == DataFormat.HDF5:
            return self._load_hdf5(data_path, symbols, start_time, end_time)
        elif data_format == DataFormat.CSV:
            return self._load_csv(data_path, symbols, start_time, end_time)
        elif data_format == DataFormat.FEATHER or data_format == DataFormat.ARROW:
            return self._load_feather(data_path, symbols, start_time, end_time)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def _detect_format(self, data_path: Path) -> DataFormat:
        """Auto-detect data format from file extension."""
        suffix = data_path.suffix.lower()
        
        format_map = {
            '.parquet': DataFormat.PARQUET,
            '.pq': DataFormat.PARQUET,
            '.h5': DataFormat.HDF5,
            '.hdf5': DataFormat.HDF5,
            '.csv': DataFormat.CSV,
            '.feather': DataFormat.FEATHER,
            '.arrow': DataFormat.ARROW,
        }
        
        detected_format = format_map.get(suffix, DataFormat.CSV)  # Default to CSV
        self.logger.info(f"Auto-detected format: {detected_format.value}")
        return detected_format
    
    def _load_parquet(self, data_path: Path, symbols: Optional[List[str]],
                     start_time: Optional[datetime], end_time: Optional[datetime]) -> Dict[str, pd.DataFrame]:
        """Load data from Parquet file."""
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
            
            # Load parquet file
            if data_path.is_file():
                # Single file
                table = pq.read_table(data_path)
                df = table.to_pandas()
            else:
                # Directory with multiple parquet files
                dataset = pq.ParquetDataset(data_path)
                table = dataset.read()
                df = table.to_pandas()
            
            return self._process_dataframe(df, symbols, start_time, end_time)
            
        except ImportError:
            self.logger.warning("PyArrow not available, falling back to pandas")
            df = pd.read_parquet(data_path)
            return self._process_dataframe(df, symbols, start_time, end_time)
    
    def _load_hdf5(self, data_path: Path, symbols: Optional[List[str]],
                  start_time: Optional[datetime], end_time: Optional[datetime]) -> Dict[str, pd.DataFrame]:
        """Load data from HDF5 file."""
        try:
            import tables  # PyTables
            
            with pd.HDFStore(data_path, mode='r') as store:
                # List available keys (datasets)
                keys = store.keys()
                self.logger.info(f"Available HDF5 datasets: {keys}")
                
                if len(keys) == 1:
                    # Single dataset
                    df = store[keys[0]]
                    return self._process_dataframe(df, symbols, start_time, end_time)
                else:
                    # Multiple datasets - assume each is a symbol
                    result = {}
                    for key in keys:
                        symbol = key.lstrip('/')  # Remove leading slash
                        if symbols is None or symbol in symbols:
                            df = store[key]
                            processed = self._process_dataframe(df, [symbol], start_time, end_time)
                            result.update(processed)
                    return result
                    
        except ImportError:
            raise ImportError("PyTables required for HDF5 support. Install with: pip install tables")
    
    def _load_csv(self, data_path: Path, symbols: Optional[List[str]],
                 start_time: Optional[datetime], end_time: Optional[datetime]) -> Dict[str, pd.DataFrame]:
        """Load data from CSV file."""
        # Try to optimize CSV reading
        try:
            # Read a sample to infer types
            sample_df = pd.read_csv(data_path, nrows=1000)
            
            # Determine dtypes
            dtype_dict = {}
            for pandas_col in sample_df.columns:
                mapped_col = self._map_column_name(pandas_col)
                if mapped_col in self.dtypes:
                    dtype_dict[pandas_col] = self.dtypes[mapped_col]
            
            # Read full file with optimized types
            df = pd.read_csv(data_path, dtype=dtype_dict)
            
        except Exception as e:
            self.logger.warning(f"Optimized CSV read failed: {e}, falling back to default")
            df = pd.read_csv(data_path)
        
        return self._process_dataframe(df, symbols, start_time, end_time)
    
    def _load_feather(self, data_path: Path, symbols: Optional[List[str]],
                     start_time: Optional[datetime], end_time: Optional[datetime]) -> Dict[str, pd.DataFrame]:
        """Load data from Feather/Arrow file."""
        df = pd.read_feather(data_path)
        return self._process_dataframe(df, symbols, start_time, end_time)
    
    def _process_dataframe(self, df: pd.DataFrame, symbols: Optional[List[str]],
                          start_time: Optional[datetime], 
                          end_time: Optional[datetime]) -> Dict[str, pd.DataFrame]:
        """Process loaded DataFrame with filtering and column mapping."""
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Parse timestamp if needed
        df = self._parse_timestamps(df)
        
        # Apply time filters
        if 'timestamp' in df.columns:
            if start_time:
                df = df[df['timestamp'] >= start_time]
            if end_time:
                df = df[df['timestamp'] <= end_time]
        
        # Split by symbol if symbol column exists
        if 'symbol' in df.columns:
            result = {}
            for symbol in df['symbol'].unique():
                if symbols is None or symbol in symbols:
                    symbol_df = df[df['symbol'] == symbol].copy()
                    symbol_df = symbol_df.sort_values('timestamp') if 'timestamp' in symbol_df.columns else symbol_df
                    result[symbol] = symbol_df
            return result
        else:
            # Single symbol data - use filename or default
            symbol = symbols[0] if symbols and len(symbols) == 1 else "UNKNOWN"
            df = df.sort_values('timestamp') if 'timestamp' in df.columns else df
            return {symbol: df}
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using mappings."""
        column_rename_map = {}
        
        for standard_name, variations in self.column_mappings.items():
            for col in df.columns:
                if col.lower() in [v.lower() for v in variations]:
                    column_rename_map[col] = standard_name
                    break
        
        if column_rename_map:
            df = df.rename(columns=column_rename_map)
            self.logger.info(f"Renamed columns: {column_rename_map}")
        
        return df
    
    def _map_column_name(self, column_name: str) -> str:
        """Map column name to standard name."""
        for standard_name, variations in self.column_mappings.items():
            if column_name.lower() in [v.lower() for v in variations]:
                return standard_name
        return column_name
    
    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp columns."""
        if 'timestamp' in df.columns:
            try:
                # Try to parse timestamps
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Ensure timezone awareness
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                
            except Exception as e:
                self.logger.warning(f"Could not parse timestamps: {e}")
        
        return df
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate loaded data and return statistics.
        
        Args:
            data: Dictionary of symbol DataFrames
            
        Returns:
            Validation results and statistics
        """
        results = {
            'symbols': list(data.keys()),
            'total_records': 0,
            'validation_errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        for symbol, df in data.items():
            symbol_stats = {
                'records': len(df),
                'columns': list(df.columns),
                'time_range': None,
                'missing_data': {},
                'data_quality': {}
            }
            
            # Count total records
            results['total_records'] += len(df)
            
            # Check for required columns
            required_cols = ['timestamp']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                results['validation_errors'].append(f"{symbol}: Missing required columns: {missing_cols}")
            
            # Check timestamp range
            if 'timestamp' in df.columns:
                try:
                    time_range = (df['timestamp'].min(), df['timestamp'].max())
                    symbol_stats['time_range'] = {
                        'start': time_range[0].isoformat(),
                        'end': time_range[1].isoformat(),
                        'duration_hours': (time_range[1] - time_range[0]).total_seconds() / 3600
                    }
                except Exception as e:
                    results['warnings'].append(f"{symbol}: Could not determine time range: {e}")
            
            # Check for missing data
            for col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    symbol_stats['missing_data'][col] = {
                        'count': int(null_count),
                        'percentage': float(null_count / len(df) * 100)
                    }
            
            # Data quality checks
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in ['price', 'bid_price', 'ask_price', 'trade_price']:
                    # Check for negative prices
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        results['validation_errors'].append(f"{symbol}: {negative_count} negative prices in {col}")
                
                elif col in ['volume', 'size', 'bid_size', 'ask_size']:
                    # Check for negative volumes
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        results['validation_errors'].append(f"{symbol}: {negative_count} negative volumes in {col}")
            
            # Check bid-ask spread if both present
            if 'bid_price' in df.columns and 'ask_price' in df.columns:
                invalid_spreads = (df['ask_price'] < df['bid_price']).sum()
                if invalid_spreads > 0:
                    results['validation_errors'].append(f"{symbol}: {invalid_spreads} invalid bid-ask spreads")
                
                # Calculate average spread
                spreads = df['ask_price'] - df['bid_price']
                symbol_stats['data_quality']['avg_spread'] = float(spreads.mean())
                symbol_stats['data_quality']['spread_std'] = float(spreads.std())
            
            results['statistics'][symbol] = symbol_stats
        
        # Overall validation result
        results['is_valid'] = len(results['validation_errors']) == 0
        
        return results
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_path: Union[str, Path],
                  data_format: DataFormat = DataFormat.PARQUET) -> None:
        """
        Save market data to file.
        
        Args:
            data: Dictionary of symbol DataFrames
            output_path: Output file path
            data_format: Output format
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if len(data) == 1:
            # Single symbol - save directly
            symbol, df = next(iter(data.items()))
            self._save_dataframe(df, output_path, data_format)
        else:
            # Multiple symbols
            if data_format == DataFormat.PARQUET:
                # Save as partitioned dataset
                combined_df = pd.concat(data.values(), ignore_index=True)
                combined_df.to_parquet(output_path, partition_cols=['symbol'] if 'symbol' in combined_df.columns else None)
            elif data_format == DataFormat.HDF5:
                # Save each symbol as separate dataset
                with pd.HDFStore(output_path, mode='w') as store:
                    for symbol, df in data.items():
                        store[f'/{symbol}'] = df
            else:
                # Save each symbol as separate file
                for symbol, df in data.items():
                    symbol_path = output_path.parent / f"{output_path.stem}_{symbol}{output_path.suffix}"
                    self._save_dataframe(df, symbol_path, data_format)
        
        self.logger.info(f"Saved data to {output_path}")
    
    def _save_dataframe(self, df: pd.DataFrame, path: Path, data_format: DataFormat) -> None:
        """Save single DataFrame to file."""
        if data_format == DataFormat.PARQUET:
            df.to_parquet(path, index=False)
        elif data_format == DataFormat.HDF5:
            df.to_hdf(path, key='data', mode='w', index=False)
        elif data_format == DataFormat.CSV:
            df.to_csv(path, index=False)
        elif data_format == DataFormat.FEATHER:
            df.to_feather(path)
        else:
            raise ValueError(f"Unsupported output format: {data_format}")
    
    def get_file_info(self, data_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about data file without loading it."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            return {'exists': False}
        
        info = {
            'exists': True,
            'path': str(data_path),
            'size_mb': data_path.stat().st_size / (1024 * 1024),
            'format': self._detect_format(data_path).value,
            'modified_time': datetime.fromtimestamp(data_path.stat().st_mtime).isoformat()
        }
        
        # Try to get more details based on format
        try:
            if info['format'] == 'parquet':
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(data_path)
                info['rows'] = pf.metadata.num_rows
                info['columns'] = len(pf.schema)
                info['schema'] = [f.name for f in pf.schema]
            
            elif info['format'] == 'hdf5':
                with pd.HDFStore(data_path, mode='r') as store:
                    info['datasets'] = store.keys()
            
        except Exception as e:
            info['details_error'] = str(e)
        
        return info
