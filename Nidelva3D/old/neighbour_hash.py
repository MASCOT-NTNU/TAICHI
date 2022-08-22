
    # def get_hash_neighbours(self):
    #     gxy = self.__waypoints[:, :2]
    #     deucli = cdist(self.__waypoints, self.__waypoints, "euclidean")
    #     if self.multiple_depth_layer:
    #         gz = self.__waypoints[:, 2].reshape(-1, 1)
    #         dg = np.abs(self.__depths[1] - self.__depths[0])
    #         dellip = (cdist(gxy, gxy, "sqeuclidean") / (1.5 * self.__neighbour_distance)**2 +
    #                   cdist(gz, gz, "sqeuclidean") / (1.5 * dg)**2)  # TODO: check a more elegant way to replace 1.5
    #     else:
    #         dellip = cdist(gxy, gxy, "sqeuclidean") / (1.5 * self.__neighbour_distance) ** 2
    #     for i in range(len(deucli)):
    #         nb_ind = np.where((dellip[i] <= 1) * (deucli[i] >= 5))[0]
    #         self.__neighbour[i] = list(nb_ind)
