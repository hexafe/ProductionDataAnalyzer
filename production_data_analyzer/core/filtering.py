import pandas as pd
from typing import Union, Optional, List, Set
from pathlib import Path
import logging
import sys

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("ProductionFilter")

class ProductionDataFilter:
    """Production data filtering with validation"""
    
    def __init__(self, production_data: pd.DataFrame) -> None:
        """
        Initialize filter system with production data

        Args:
            production_data: Complete manufacturing records as DataFrame

        Raises:
            TypeError: If input is not a pandas DataFrame
        """
        if not isinstance(production_data, pd.DataFrame):
            raise TypeError("Input must be pandas DataFrame")
        self.original_data = production_data
        self.filtered_data: Optional[pd.DataFrame] = None
        self.filter_log: List[dict] = []
        logger.info(f"Initialized ProductionFilter with {len(production_data):,} records")

    def filter_by_id(
        self,
        reference_ids: Union[pd.DataFrame, List, Set],
        id_column: str,
        strict_mode: bool = True,
        persist_filtered: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Filter production data using reference IDs with comprehensive validation

        Args:
            reference_ids: Valid IDs to filter by. Can be:
                - DataFrame with ID column
                - List/Sets of ID values
            id_column: Column name containing IDs in production data
            strict_mode: If True, raises errors on validation issues
                If False, returns None and logs warnings
            persist_filtered: Maintain filtered copy in instance state

        Returns:
            Filtered DataFrame if successful, None if failed in non-strict mode

        Raises:
            KeyError: Missing ID column in production or reference data
            TypeError: Invalid reference_ids type
            ValueError: Data validation failures (empty IDs, conversion issues)
        """
        try:
            self._validate_filter_inputs(reference_ids, id_column)
            
            # Convert IDs to normalized strings
            prod_ids = self._safe_id_conversion(
                self.original_data[id_column], 
                context="production"
            )
            ref_ids = self._safe_id_conversion(
                self._get_reference_series(reference_ids, id_column),
                context="reference"
            )

            # Apply filter and handle results
            mask = prod_ids.isin(ref_ids)
            filtered = self.original_data[mask].copy()
            
            if persist_filtered:
                self.filtered_data = filtered
                self._log_filter_operation(id_column, mask.sum())
                logger.info(f"Filter persisted: {len(filtered):,} records")

            return filtered

        except Exception as e:
            if strict_mode:
                raise
            logger.warning(f"Filtering aborted: {str(e)}")
            return None

    def export_filter_report(self, output_path: Path) -> None:
        """
        Export filtering metadata and statistics to CSV

        Args:
            output_path: Destination path for audit report

        Raises:
            ValueError: If no filtering operations logged
            OSError: For filesystem errors during save
        """
        if not self.filter_log:
            raise ValueError("No filtering operations to report")

        try:
            report = pd.DataFrame(self.filter_log)
            report.to_csv(output_path, index=False)
            logger.info(f"Audit report saved to {output_path}")
        except OSError as e:
            logger.error(f"Failed to save report: {str(e)}")
            raise

    def _validate_filter_inputs(
        self,
        reference_ids: Union[pd.DataFrame, List, Set],
        id_column: str
    ) -> None:
        """
        Validate filtering inputs

        Args:
            reference_ids: Reference IDs to validate
            id_column: ID column name to check

        Raises:
            KeyError: Missing ID column
            TypeError: Invalid reference_ids type
        """
        # Production data validation
        if id_column not in self.original_data.columns:
            raise KeyError(f"ID column '{id_column}' missing from production data")
            
        # Reference data validation
        if isinstance(reference_ids, pd.DataFrame):
            if id_column not in reference_ids.columns:
                raise KeyError(f"ID column '{id_column}' missing from reference DataFrame")
        elif not isinstance(reference_ids, (list, set)):
            raise TypeError("Reference IDs must be DataFrame, list or set")

    def _get_reference_series(
        self,
        reference_ids: Union[pd.DataFrame, List, Set],
        id_column: str
    ) -> pd.Series:
        """
        Convert reference IDs to pandas Series

        Args:
            reference_ids: Input reference IDs
            id_column: ID column name for DataFrame inputs

        Returns:
            Unified Series of reference IDs
        """
        if isinstance(reference_ids, pd.DataFrame):
            return reference_ids[id_column]
        return pd.Series(list(reference_ids))

    def _safe_id_conversion(
        self,
        series: pd.Series,
        context: str = "data"
    ) -> pd.Series:
        """
        Convert and validate ID series to normalized strings

        Args:
            series: Raw ID data to convert
            context: Context for error messages (production/reference)

        Returns:
            Normalized ID strings

        Raises:
            ValueError: Conversion failures or empty IDs
        """
        try:
            # Convert to strings and normalize
            converted = series.astype(str).str.strip()
            
            # Check for empty values
            if converted.str.contains(r'^\s*$').any():
                raise ValueError(f"Empty IDs detected in {context} data")
                
            # Check for conversion artifacts
            numeric_check = pd.to_numeric(converted, errors='coerce')
            if (numeric_check.notnull() & (converted != numeric_check.astype(str))).any():
                logger.warning(f"Potential ID conversion artifacts in {context} data")

            return converted

        except Exception as e:
            raise ValueError(f"ID conversion failed for {context} data: {str(e)}") from e

    def _log_filter_operation(
        self,
        id_column: str,
        kept_rows: int
    ) -> None:
        """
        Record filtering operation metadata

        Args:
            id_column: ID column used for filtering
            kept_rows: Number of retained records
        """
        log_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'id_column': id_column,
            'original_rows': len(self.original_data),
            'filtered_rows': kept_rows,
            'retention_pct': round((kept_rows / len(self.original_data)) * 100, 2),
            'filter_type': 'ID-based'
        }
        self.filter_log.append(log_entry)
        logger.debug(f"Logged filter operation: {log_entry}")
