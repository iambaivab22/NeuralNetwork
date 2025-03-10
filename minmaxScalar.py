class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        """Compute the minimum and maximum values for scaling."""
        self.min = min(data)
        self.max = max(data)

    def transform(self, data):
        """Scale the data based on the computed min and max."""
        if self.min is None or self.max is None:
            raise ValueError("Scaler has not been fitted yet. Call `fit` first.")
        return [(x - self.min) / (self.max - self.min) for x in data]

    def fit_transform(self, data):
        """Fit and transform the data in one step."""
        self.fit(data)
        return self.transform(data)


# Example usage
if __name__ == "__main__":
    # Sample data
    data = [10, 20, 30, 40, 50]

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler and transform the data
    scaled_data = scaler.fit_transform(data)

    print("Original Data for minmax scalar:", data)
    print("Scaled Data:", scaled_data)