from torch.utils.data.sampler import BatchSampler


class PlayerSampler(BatchSampler):
    """
    Custom PyTorch Sampler that limits the number of frames per player.
    """

    def __init__(
        self, sampler, players, batch_size=32, drop_last=False, samples_per_player=10
    ):
        """
        Constructor.

        Args:
            sampler (torch sampler): Initial sampler for the dataset, e.g. RandomSampler
            players (numpy array): Player corresponding to each sample. Precomputed to gain time.
            batch_size (int, optional): Batch size. Defaults to 32.
            drop_last (bool, optional): Whether to discard the last batch. Defaults to False.
            samples_per_player (int, optional): Maximum of image to use per id. Defaults to 10.
        """
        super().__init__(sampler, batch_size, drop_last)
        self.samples_per_player = samples_per_player
        self.players = players

        self.len = self.compute_len()

    def __len__(self):
        return self.len

    def compute_len(self):
        player_counts = {}
        yielded = 0
        batch = []

        for idx in self.sampler:
            player = self.players[idx]
            try:
                player_counts[player] += 1
            except KeyError:
                player_counts[player] = 1

            if player_counts[player] <= self.samples_per_player:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    yielded += 1
                    batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1

        return yielded

    def __iter__(self):
        """
        Iterator.
        Only adds an index to a batch if the associated players has not be sampled too many time.

        Yields:
            torch tensors : batches.
        """
        player_counts = {}
        yielded = 0
        batch = []

        for idx in self.sampler:
            player = self.players[idx]
            try:
                player_counts[player] += 1
            except KeyError:
                player_counts[player] = 1

            if player_counts[player] <= self.samples_per_player:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    yield batch
                    yielded += 1
                    batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch
