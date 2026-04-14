import matplotlib

# Use the Agg backend in pytest to avoid interactive display requirements on CI / remote machines.
matplotlib.use("Agg")
