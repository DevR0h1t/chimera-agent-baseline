# Low VRAM tier — edit to match your cluster
# Run `sinfo -N -l` to see available nodes.
NODELIST=dlc-tornadus,dlc-articuno,dlc-lugia,dlc-moltres,dlc-nidoking,dlc-zapdos,dlc-mewtwo
GRES=gpu:1
CPUS=8
MEM=32G
TIME=24:00:00
