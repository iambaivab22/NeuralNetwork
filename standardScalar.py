class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """Compute the mean and standard deviation for scaling."""
        n = len(data)
        self.mean = sum(data) / n
        variance = sum((x - self.mean) ** 2 for x in data) / n
        self.std = variance ** 0.5  # Standard deviation

    def transform(self, data):
        """Scale the data based on the computed mean and standard deviation."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet. Call `fit` first.")
        if self.std == 0:
            raise ValueError("Standard deviation is zero. Cannot scale data.")
        return [(x - self.mean) / self.std for x in data]

    def fit_transform(self, data):
        """Fit and transform the data in one step."""
        self.fit(data)
        return self.transform(data)


# Example usage
if __name__ == "__main__":
    # Sample data
    data = [10, 20, 30, 40, 50]

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler and transform the data
    scaled_data = scaler.fit_transform(data)

    print("Original Data:", data)
    print("Scaled Data:", scaled_data)